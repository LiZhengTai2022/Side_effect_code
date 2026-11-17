#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
    BaggingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import joblib

INPUT_FILE = "side_with_smiles.xlsx"
MODEL_SAVE_DIR = "side_effect_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
MAX_SEQ_LEN = 1024
TOP_N_CLASSES = 8

N_FOLDS = 5
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3
PATIENCE = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_morgan_fingerprint(smiles, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=np.float32)
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits).GetFingerprint(mol)
    return np.array(fp, dtype=np.float32)


def get_aa_composition(sequence, max_len=MAX_SEQ_LEN):
    if pd.isna(sequence) or not isinstance(sequence, str):
        return np.zeros(25, dtype=np.float32)

    seq = sequence[:max_len]
    aa_list = "ACDEFGHIKLMNPQRSTVWY"
    
    composition = np.array([seq.count(aa) / len(seq) for aa in aa_list], dtype=np.float32)
    
    stats = np.array([
        len(seq) / max_len,
        sum(1 for c in seq if c in 'GAVLIM') / len(seq),
        sum(1 for c in seq if c in 'DEKRH') / len(seq),
        sum(1 for c in seq if c in 'STYQN') / len(seq),
        sum(1 for c in seq if c == 'C') / len(seq),
    ], dtype=np.float32)
    
    return np.concatenate([composition, stats])


def build_features(df):
    print("  [1/4] 提取药物 Morgan 指纹...")
    drug_fps = np.array([get_morgan_fingerprint(s) for s in df['SMILES']], dtype=np.float32)
    
    print("  [2/4] 提取蛋白序列特征...")
    prot_feats = np.array([get_aa_composition(s) for s in df['Protein_Sequence']], dtype=np.float32)
    
    print("  [3/4] 提取结合分数和方向...")
    affinity = df['Predicted_Affinity'].fillna(df['Predicted_Affinity'].median()).values.reshape(-1, 1).astype(np.float32)
    
    direction_map = {'increase': 1, 'decrease': 0}
    direction = df['变化方向'].map(direction_map).fillna(0).values.reshape(-1, 1).astype(np.float32)
    
    print("  [4/4] 合并特征...")
    X = np.hstack([drug_fps, prot_feats, affinity, direction])
    print(f"  特征维度: {X.shape[1]} (Morgan:{drug_fps.shape[1]} + Protein:{prot_feats.shape[1]} + Affinity:1 + Direction:1)")
    
    return X


class SideEffectDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
        return x, y


class SideEffectNet(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.4):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.classifier = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.dropout(F.gelu(self.bn1(self.fc1(x))))
        x = self.dropout(F.gelu(self.bn2(self.fc2(x))))
        x = self.dropout(F.gelu(self.bn3(self.fc3(x))))
        x = self.dropout(F.gelu(self.bn4(self.fc4(x))))
        return self.classifier(x)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def train_deep_model(X_train, y_train, X_val, y_val, num_classes, input_dim):
    train_dataset = SideEffectDataset(X_train, y_train, augment=True)
    val_dataset = SideEffectDataset(X_val, y_val, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SideEffectNet(input_dim, num_classes).to(DEVICE)
    criterion = LabelSmoothingLoss(num_classes, smoothing=0.1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_acc = 0
    best_state = None
    no_improve = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                output = model(batch_X)
                val_preds.extend(output.argmax(dim=1).cpu().numpy())
                val_true.extend(batch_y.numpy())
        
        val_acc = accuracy_score(val_true, val_preds)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
        
        if no_improve >= PATIENCE:
            print(f"    早停于 Epoch {epoch+1}，最佳 Val Acc: {best_acc:.4f}")
            break
    
    model.load_state_dict(best_state)
    return model, best_acc


def main():
    print("=" * 70)
    print("副作用多分类预测模型训练（优化版）")
    print("=" * 70)
    
    print(f"\n[1/6] 读取数据: {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE)
    print(f"  原始数据: {len(df)} 行")
    
    valid_mask = (
        df['SMILES'].notna() & 
        df['Protein_Sequence'].notna() & 
        df['副作用分类(英文)'].notna()
    )
    df = df[valid_mask].reset_index(drop=True)
    print(f"  有效数据: {len(df)} 行")
    
    print(f"\n[2/6] 筛选前 {TOP_N_CLASSES} 大类别...")
    top_classes = df['副作用分类(英文)'].value_counts().head(TOP_N_CLASSES).index.tolist()
    print(f"  保留类别: {top_classes}")
    
    df = df[df['副作用分类(英文)'].isin(top_classes)].reset_index(drop=True)
    print(f"  筛选后数据: {len(df)} 行")
    
    le = LabelEncoder()
    y = le.fit_transform(df['副作用分类(英文)'])
    num_classes = len(le.classes_)
    print(f"  类别数: {num_classes}")
    print(f"  类别分布:")
    for cls, count in zip(le.classes_, np.bincount(y)):
        print(f"    {cls}: {count}")
    
    print(f"\n[3/6] 构建特征矩阵...")
    X = build_features(df)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"\n[4/6] 训练传统机器学习模型 (5折交叉验证)...")
    
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_STATE, eval_metric='mlogloss',
            tree_method='hist'
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=RANDOM_STATE, verbose=-1,
            force_col_wise=True
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_split=5,
            min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_split=5,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_STATE
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE
        ),
        'LogisticRegression': LogisticRegression(
            max_iter=2000, C=1.0, multi_class='multinomial',
            random_state=RANDOM_STATE, solver='lbfgs'
        ),
        'SVM': SVC(
            kernel='rbf', C=10, gamma='scale', random_state=RANDOM_STATE,
            probability=True
        ),
    }
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    
    for name, model in models.items():
        print(f"\n  训练 {name}...")
        try:
            scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
            mean_acc = scores.mean()
            std_acc = scores.std()
        except Exception as e:
            print(f"    异常: {e}")
            mean_acc = 0.0
            std_acc = 0.0
            scores = [0.0]
        
        results[name] = {
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'scores': scores
        }
        print(f"    准确率: {mean_acc:.4f} (+/- {std_acc:.4f})")
    
    print(f"\n[5/6] 训练深度学习模型...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    dl_model, dl_acc = train_deep_model(X_train, y_train, X_val, y_val, num_classes, X.shape[1])
    results['DeepLearning'] = {'mean_acc': dl_acc, 'std_acc': 0, 'scores': [dl_acc]}
    
    print(f"\n[6/6] 选择最佳模型并保存...")
    
    valid_results = {k: v for k, v in results.items() if not np.isnan(v['mean_acc'])}
    best_name = max(valid_results, key=lambda k: valid_results[k]['mean_acc'])
    
    print(f"\n  最佳模型: {best_name} (准确率: {valid_results[best_name]['mean_acc']:.4f})")
    
    print("\n  所有模型对比:")
    print(f"  {'模型':<20} {'平均准确率':<12} {'标准差':<10}")
    print(f"  {'-'*42}")
    for name, res in sorted(valid_results.items(), key=lambda x: x[1]['mean_acc'], reverse=True):
        print(f"  {name:<20} {res['mean_acc']:<12.4f} {res['std_acc']:<10.4f}")
    
    print(f"\n  生成模型对比柱状图...")
    
    sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['mean_acc'], reverse=True)
    model_names = [name for name, _ in sorted_models]
    model_accs = [res['mean_acc'] for _, res in sorted_models]
    model_stds = [res['std_acc'] for _, res in sorted_models]
    
    colors = ['#2196F3'] * len(model_names)
    colors[0] = '#FF5722'
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(model_names, model_accs, xerr=model_stds, color=colors, 
                   edgecolor='white', linewidth=1.5, capsize=5, error_kw={'elinewidth': 2})
    
    for bar, acc in zip(bars, model_accs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{acc:.4f}', va='center', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Model Comparison - Side Effect Classification', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(0, max(model_accs) * 1.15)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    
    chart_path = os.path.join(MODEL_SAVE_DIR, "model_comparison.png")
    plt.savefig(chart_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  柱状图已保存: {chart_path}")
    
    print(f"  生成类别分布饼图...")
    
    class_names = le.classes_
    class_counts = np.bincount(y)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    explode = [0.05] * len(class_names)
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    wedges, texts, autotexts = axes[0].pie(class_counts, explode=explode, labels=class_names,
                                            autopct='%1.1f%%', colors=colors_pie,
                                            startangle=90, textprops={'fontsize': 10})
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    axes[0].set_title('Class Distribution', fontsize=13, fontweight='bold')
    
    bars = axes[1].bar(range(len(class_names)), class_counts, color=colors_pie, edgecolor='white', linewidth=1.5)
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[1].set_title('Class Sample Count', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_facecolor('#f8f9fa')
    
    for bar, count in zip(bars, class_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     str(count), ha='center', fontweight='bold', fontsize=10)
    
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    
    class_chart_path = os.path.join(MODEL_SAVE_DIR, "class_distribution.png")
    plt.savefig(class_chart_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  类别分布图已保存: {class_chart_path}")
    
    if best_name == 'DeepLearning':
        print(f"\n  使用深度学习模型进行全量预测...")
        dl_model_full, _ = train_deep_model(X, y, X, y, num_classes, X.shape[1])
        dl_model_full.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(DEVICE)
            y_pred = dl_model_full(X_tensor).argmax(dim=1).cpu().numpy()
    else:
        best_model = models[best_name]
        best_model.fit(X, y)
        y_pred = best_model.predict(X)
    print(f"\n  全量训练集分类报告:")
    print(classification_report(y, y_pred, target_names=le.classes_))
    
    for name, model in models.items():
        if name in valid_results:
            try:
                model_clone = type(model)(**model.get_params())
                model_clone.fit(X, y)
                joblib.dump(model_clone, os.path.join(MODEL_SAVE_DIR, f"model_{name}.pkl"))
            except:
                pass
    
    joblib.dump(le, os.path.join(MODEL_SAVE_DIR, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "scaler.pkl"))
    
    torch.save({
        'model_state_dict': dl_model.state_dict(),
        'input_dim': X.shape[1],
        'num_classes': num_classes,
    }, os.path.join(MODEL_SAVE_DIR, "best_deep_model.pth"))
    
    print(f"\n  模型已保存到: {MODEL_SAVE_DIR}/")
    print(f"    - model_*.pkl (所有传统模型)")
    print(f"    - best_deep_model.pth")
    print(f"    - label_encoder.pkl")
    print(f"    - scaler.pkl")
    
    df['Predicted_SideEffect'] = le.inverse_transform(y_pred)
    df['Predicted_SideEffect_Code'] = y_pred
    
    df['Correct'] = (y_pred == y)
    
    output_file = INPUT_FILE
    df.to_excel(output_file, index=False)
    print(f"\n  预测结果已写入: {output_file}")
    
    print(f"\n  生成预测结果汇总表...")
    
    precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, labels=range(num_classes))
    correct_per_class = np.array([np.sum((y_pred == c) & (y == c)) for c in range(num_classes)])
    total_per_class = np.bincount(y, minlength=num_classes)
    acc_per_class = np.where(total_per_class > 0, correct_per_class / total_per_class, 0)
    
    summary_df = pd.DataFrame({
        'Side Effect Category': le.classes_,
        'Sample Count': total_per_class,
        'Correct Predictions': correct_per_class,
        'Accuracy': np.round(acc_per_class, 4),
        'Precision': np.round(precision, 4),
        'Recall': np.round(recall, 4),
        'F1-Score': np.round(f1, 4),
    })
    
    overall_row = pd.DataFrame({
        'Side Effect Category': ['Overall / Weighted Avg'],
        'Sample Count': [len(y)],
        'Correct Predictions': [np.sum(y_pred == y)],
        'Accuracy': [np.round(accuracy_score(y, y_pred), 4)],
        'Precision': [np.round(np.average(precision, weights=support), 4)],
        'Recall': [np.round(np.average(recall, weights=support), 4)],
        'F1-Score': [np.round(np.average(f1, weights=support), 4)],
    })
    summary_df = pd.concat([summary_df, overall_row], ignore_index=True)
    
    summary_path = os.path.join(MODEL_SAVE_DIR, "prediction_summary.xlsx")
    summary_df.to_excel(summary_path, index=False)
    print(f"  预测汇总表已保存: {summary_path}")
    
    print(f"\n  预测结果汇总表:")
    print(summary_df.to_string(index=False))
    
    print(f"\n  生成各分类详细指标图...")
    
    x = np.arange(num_classes)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#4CAF50', edgecolor='white')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2196F3', edgecolor='white')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#FF9800', edgecolor='white')
    
    ax.set_xlabel('Side Effect Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    metrics_chart_path = os.path.join(MODEL_SAVE_DIR, "per_class_metrics.png")
    plt.savefig(metrics_chart_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  分类指标图已保存: {metrics_chart_path}")
    
    print(f"\n{'=' * 70}")
    print("训练完成!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()