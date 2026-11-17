#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

MODEL_DIR = "side_effect_models"
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
MAX_SEQ_LEN = 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_morgan_fingerprint(smiles, radius=MORGAN_RADIUS, nBits=MORGAN_NBITS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=np.float32)
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits).GetFingerprint(mol)
    return np.array(fp, dtype=np.float32)


def get_aa_composition(sequence, max_len=MAX_SEQ_LEN):
    if pd.isna(sequence) or not isinstance(sequence, str) or not sequence:
        return np.zeros(25, dtype=np.float32)

    seq = sequence[:max_len]
    aa_list = "ACDEFGHIKLMNPQRSTVWY"

    composition = np.array([seq.count(aa) / len(seq) for aa in aa_list], dtype=np.float32)

    stats = np.array([
        len(seq) / max_len,
        sum(c in 'GAVLIM' for c in seq) / len(seq),
        sum(c in 'DEKRH' for c in seq) / len(seq),
        sum(c in 'STYQN' for c in seq) / len(seq),
        seq.count('C') / len(seq),
    ], dtype=np.float32)

    return np.concatenate([composition, stats])


def build_features(smiles, protein_seq, affinity, direction):
    drug_fp = get_morgan_fingerprint(smiles).reshape(1, -1)
    prot_feat = get_aa_composition(protein_seq).reshape(1, -1)
    aff_arr = np.array([[affinity]], dtype=np.float32)
    dir_arr = np.array([[direction]], dtype=np.float32)

    return np.hstack([drug_fp, prot_feat, aff_arr, dir_arr])


def predict_affinity(smiles, protein_seq):
    model_path = "best_model-0.9470.pth"
    char_map_path = "char_to_int.pt"
    scaler_path = "scaler.pkl"

    if not (os.path.exists(model_path) and os.path.exists(char_map_path)):
        print("  结合模型文件未找到，使用默认值")
        return None

    try:
        from torch_geometric.data import Data, Batch
        from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
        from torch.nn.utils.rnn import pad_sequence
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        from rdkit.Chem import rdmolops

        char_to_int = torch.load(char_map_path, map_location="cpu")
        num_protein_features = len(char_to_int) + 1

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        atom_feats = [[atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetTotalDegree()] for atom in mol.GetAtoms()]
        x = torch.tensor(atom_feats, dtype=torch.float)
        adj = rdmolops.GetAdjacencyMatrix(mol)
        edge_index = torch.tensor(np.nonzero(adj), dtype=torch.long)
        comp_data = Data(x=x, edge_index=edge_index)

        seq_enc = [char_to_int.get(c, 0) for c in protein_seq]
        seq_enc = seq_enc[:MAX_SEQ_LEN] if len(seq_enc) > MAX_SEQ_LEN else seq_enc + [0] * (MAX_SEQ_LEN - len(seq_enc))
        seq_tensor = torch.tensor(seq_enc, dtype=torch.long).unsqueeze(0)

        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, dropout=0.1, max_len=MAX_SEQ_LEN):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)
                pe = torch.zeros(1, max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
                pe[0, :, 0::2] = torch.sin(position * div_term)
                pe[0, :, 1::2] = torch.cos(position * div_term)
                self.register_buffer('pe', pe)
            def forward(self, x):
                return self.dropout(x + self.pe[:, :x.size(1), :])

        class AttentionPooling(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.attention = nn.Sequential(
                    nn.Linear(dim, dim // 2),
                    nn.Tanh(),
                    nn.Linear(dim // 2, 1)
                )
            def forward(self, x, mask=None):
                scores = self.attention(x).squeeze(-1)
                if mask is not None:
                    scores = scores.masked_fill(mask, float('-inf'))
                weights = F.softmax(scores, dim=-1).unsqueeze(-1)
                return torch.sum(x * weights, dim=1)

        class BindingModel(nn.Module):
            def __init__(self, node_feat_dim, prot_feat_dim, model_dim=256, heads=8, dropout=0.4):
                super().__init__()
                self.conv1 = GATv2Conv(node_feat_dim, model_dim, heads=4, concat=True, dropout=0.1)
                self.bn1 = nn.BatchNorm1d(model_dim * 4)
                self.conv2 = GATv2Conv(model_dim * 4, model_dim, heads=4, concat=True, dropout=0.1)
                self.bn2 = nn.BatchNorm1d(model_dim * 4)
                self.conv3 = GATv2Conv(model_dim * 4, model_dim, heads=1, concat=False, dropout=0.1)

                self.embedding = nn.Embedding(prot_feat_dim, model_dim)
                self.pos_encoder = PositionalEncoding(model_dim)
                encoder_layer = TransformerEncoderLayer(
                    d_model=model_dim, nhead=heads, dim_feedforward=model_dim*4, dropout=dropout,
                    activation='gelu', batch_first=True, norm_first=True
                )
                self.transformer = TransformerEncoder(encoder_layer, num_layers=2)

                self.prot_pool = AttentionPooling(model_dim)
                self.cross_attn = nn.MultiheadAttention(embed_dim=model_dim*2, num_heads=heads, batch_first=True)
                self.norm_fusion = nn.LayerNorm(model_dim*2)

                self.interaction_mlp = nn.Sequential(
                    nn.Linear(model_dim*2, model_dim*4),
                    nn.GELU(),
                    nn.Linear(model_dim*4, model_dim*2),
                    nn.Dropout(dropout)
                )

                self.final_mlp = nn.Sequential(
                    nn.Linear(model_dim * 5, model_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(model_dim * 2, 1)
                )

            def forward(self, comp_data, seq_data):
                x, edge, batch = comp_data.x, comp_data.edge_index, comp_data.batch
                x = F.elu(self.bn1(self.conv1(x, edge)))
                x = F.elu(self.bn2(self.conv2(x, edge)))
                x_atom = F.elu(self.conv3(x, edge))

                comp_pool = torch.cat([global_mean_pool(x_atom, batch), global_max_pool(x_atom, batch)], dim=1)

                mask = (seq_data == 0)
                seq_emb = self.embedding(seq_data)
                seq_enc = self.transformer(self.pos_encoder(seq_emb), src_key_padding_mask=mask)
                prot_pool = self.prot_pool(seq_enc, mask=mask)

                query = comp_pool.unsqueeze(1)
                kv = torch.cat([seq_enc, seq_enc], dim=-1)
                attn_out, _ = self.cross_attn(query, kv, kv, key_padding_mask=mask)

                inter = self.interaction_mlp(self.norm_fusion(comp_pool + attn_out.squeeze(1)))

                out = self.final_mlp(torch.cat([comp_pool, prot_pool, inter], dim=-1))
                return out

        model = BindingModel(3, num_protein_features, model_dim=256, heads=8, dropout=0.4).to(DEVICE)

        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        with torch.no_grad():
            batch = Batch.from_data_list([comp_data]).to(DEVICE)
            seq = seq_tensor.to(DEVICE)
            output = model(batch, seq)
            affinity_scaled = output.squeeze().cpu().item()

        affinity_orig = affinity_scaled
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                affinity_orig = scaler.inverse_transform([[affinity_scaled]])[0][0]
                print(f"  结合亲和力: {affinity_orig:.4f} (标准化: {affinity_scaled:.4f})")
            except Exception as e:
                print(f"  scaler 反标准化失败: {e}，使用标准化值")
        else:
            print(f"  结合亲和力 (标准化值): {affinity_scaled:.4f}")

        return affinity_orig

    except Exception as e:
        print(f"  结合模型预测失败: {e}")
        return None


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

    def forward(self, x):
        x = self.dropout(F.gelu(self.bn1(self.fc1(x))))
        x = self.dropout(F.gelu(self.bn2(self.fc2(x))))
        x = self.dropout(F.gelu(self.bn3(self.fc3(x))))
        x = self.dropout(F.gelu(self.bn4(self.fc4(x))))
        return self.classifier(x)


def analyze_causal_chain(smiles, protein_seq, direction, affinity, model, scaler, le, device):
    dir_val = 1 if direction.lower() == 'increase' else 0
    X_orig = build_features(smiles, protein_seq, affinity, dir_val)
    X_scaled = scaler.transform(X_orig)

    drug_start, drug_end = 0, 2048
    prot_start, prot_end = 2048, 2073
    aff_idx = 2073
    dir_idx = 2074

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        output = model(X_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        pred_label = le.inverse_transform([pred_idx])[0]
        base_prob = probs[pred_idx]

    contribs = {}

    def perturb_and_get_contrib(idx_start, idx_end=None, single_idx=None, name=None):
        X_pert = X_scaled.copy()
        if single_idx is not None:
            X_pert[0, single_idx] = 0
        elif idx_end is not None:
            X_pert[0, idx_start:idx_end] = 0
        with torch.no_grad():
            out = model(torch.FloatTensor(X_pert).to(device))
            prob = F.softmax(out, dim=1).cpu().numpy()[0][pred_idx]
        contribs[name] = base_prob - prob

    perturb_and_get_contrib(drug_start, drug_end, name='药物分子结构')
    perturb_and_get_contrib(prot_start, prot_end, name='菌种蛋白特征')
    perturb_and_get_contrib(None, None, aff_idx, name='结合亲和力')
    perturb_and_get_contrib(None, None, dir_idx, name='菌种变化方向')

    total = sum(abs(v) for v in contribs.values())
    if total > 0:
        contrib_pct = {k: (v / total) * 100 for k, v in contribs.items()}
    else:
        contrib_pct = {k: 25.0 for k in contribs}

    mol = Chem.MolFromSmiles(smiles)
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_NBITS).GetFingerprint(mol)
    active_bits = [i for i in range(MORGAN_NBITS) if fp[i] == 1]

    aa_list = "ACDEFGHIKLMNPQRSTVWY"
    seq = protein_seq[:MAX_SEQ_LEN]
    aa_comp = {aa: seq.count(aa) / len(seq) * 100 for aa in aa_list if seq.count(aa) > 0}
    aa_top = sorted(aa_comp.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        'predicted_class': pred_label,
        'confidence': base_prob,
        'contributions': contrib_pct,
        'active_fingerprint_bits': len(active_bits),
        'top_amino_acids': aa_top,
        'affinity': affinity,
        'direction': direction,
        'all_probs': {cls: float(prob) for cls, prob in zip(le.classes_, probs)},
    }


def print_causal_chain(analysis):
    print()
    print("副作用因果链分析")
    print()
    print(f"  预测副作用: {analysis['predicted_class']}")
    print(f"  置信度: {analysis['confidence']:.2%}")
    print()

    print(f"  药物分子结构  →  {analysis['active_fingerprint_bits']} 个活性子结构")
    print("       ↓")
    print(f"  结合亲和力: {analysis['affinity']:.4f}")
    print("       ↓")
    print("  菌种蛋白特征  →  Top 5 氨基酸:")
    for aa, pct in analysis['top_amino_acids']:
        print(f"    {aa}: {pct:.1f}%")
    print("       ↓")
    print(f"  菌种变化方向: {analysis['direction']}")
    print("       ↓")
    print(f"  副作用: {analysis['predicted_class']} ({analysis['confidence']:.2%})")
    print()

    print("  特征贡献度分析:")
    sorted_contrib = sorted(analysis['contributions'].items(), key=lambda x: x[1], reverse=True)
    for feat, c in sorted_contrib:
        length = max(1, int(abs(c) / 2))
        bar = '█' * length
        sign = '+' if c > 0 else ''
        print(f"    {feat:<15} {sign}{c:.1f}%  {bar}")
    print()

    print("  所有类别概率:")
    sorted_probs = sorted(analysis['all_probs'].items(), key=lambda x: x[1], reverse=True)
    for cls, p in sorted_probs:
        bar = '█' * int(p * 30)
        marker = " ← 预测结果" if cls == analysis['predicted_class'] else ""
        print(f"    {cls:<25} {p:.2%}  {bar}{marker}")
    print("=" * 70)


def predict_side_effect(smiles, protein_seq, direction, return_analysis=False):
    label_enc_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    if not os.path.exists(label_enc_path):
        raise FileNotFoundError("未找到 label_encoder.pkl，请先运行训练脚本")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("未找到 scaler.pkl，请先运行训练脚本")

    le = joblib.load(label_enc_path)
    scaler = joblib.load(scaler_path)

    print("  预测结合亲和力...")
    affinity = predict_affinity(smiles, protein_seq)
    if affinity is None:
        print("  结合模型不可用，使用默认亲和力值 (中位值)")
        affinity = 0.0

    print("  构建特征向量...")
    dir_val = 1 if direction.lower() == 'increase' else 0
    X = build_features(smiles, protein_seq, affinity, dir_val)
    X = scaler.transform(X)

    model_path = os.path.join(MODEL_DIR, "best_deep_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到深度学习模型: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = SideEffectNet(checkpoint['input_dim'], checkpoint['num_classes']).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        output = model(X_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)

    pred_label = le.inverse_transform([pred_idx])[0]

    result = {
        'predicted_class': pred_label,
        'predicted_code': pred_idx,
        'probabilities': {cls: float(prob) for cls, prob in zip(le.classes_, probs)},
        'top_3': sorted(((cls, float(prob)) for cls, prob in zip(le.classes_, probs)), key=lambda x: x[1], reverse=True)[:3],
        'affinity_used': affinity,
        'direction_used': dir_val
    }

    if return_analysis:
        print("  进行因果链分析...")
        analysis = analyze_causal_chain(smiles, protein_seq, direction, affinity, model, scaler, le, DEVICE)
        result['causal_analysis'] = analysis

    return result


def interactive_mode():
    print("=" * 60)
    print("副作用预测系统")
    print("=" * 60)
    print()

    while True:
        print("-" * 60)
        print("请输入以下信息（输入 'q' 退出）:")
        print()

        smiles = input("1. 药物 SMILES 结构: ").strip()
        if smiles.lower() == 'q':
            print("退出预测系统")
            break
        if not smiles:
            print("  错误: SMILES 不能为空")
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("  错误: 无效的 SMILES 格式，请检查后重试")
            continue

        protein_seq = input("2. 菌群蛋白序列: ").strip()
        if protein_seq.lower() == 'q':
            print("退出预测系统")
            break
        if not protein_seq:
            print("  错误: 蛋白序列不能为空")
            continue

        print("3. 菌群变化方向:")
        print("   [1] increase (上升)")
        print("   [2] decrease (下降)")
        dir_choice = input("   请选择 (1/2): ").strip()
        if dir_choice.lower() == 'q':
            print("退出预测系统")
            break
        if dir_choice == '1':
            direction = 'increase'
        elif dir_choice == '2':
            direction = 'decrease'
        else:
            print("  错误: 请输入 1 或 2")
            continue

        print()
        print("正在预测...")
        try:
            res = predict_side_effect(smiles, protein_seq, direction, return_analysis=True)

            print()
            print("=" * 60)
            print("预测结果")
            print("=" * 60)
            print(f"  预测副作用: {res['predicted_class']}")
            print(f"  置信度: {res['probabilities'][res['predicted_class']]:.2%}")
            print()
            print("  Top 3 预测:")
            for i, (cls, prob) in enumerate(res['top_3'], 1):
                bar = '█' * int(prob * 30)
                print(f"    {i}. {cls:<25} {prob:.2%} {bar}")
            print()
            print(f"  使用结合亲和力: {res['affinity_used']:.4f}")
            print(f"  使用变化方向: {'increase' if res['direction_used'] == 1 else 'decrease'}")
            print("=" * 60)

            if 'causal_analysis' in res:
                print_causal_chain(res['causal_analysis'])

        except Exception as e:
            print(f"  预测失败: {e}")

        print()
        cont = input("是否继续预测? (y/n): ").strip().lower()
        if cont != 'y':
            print("退出预测系统")
            break
        print()


def batch_mode(input_file, output_file=None):
    if output_file is None:
        output_file = input_file.replace('.xlsx', '_predicted.xlsx')

    print(f"读取数据: {input_file}")
    df = pd.read_excel(input_file)

    for col in ['SMILES', 'Protein_Sequence', '变化方向']:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    print(f"共 {len(df)} 条数据，开始批量预测...")

    preds = []
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        protein_seq = row['Protein_Sequence']
        direction = row['变化方向']

        if pd.isna(smiles) or pd.isna(protein_seq):
            preds.append(None)
            continue

        try:
            res = predict_side_effect(smiles, protein_seq, direction)
            preds.append(res['predicted_class'])
        except Exception as e:
            print(f"  第 {idx+1} 条预测失败: {e}")
            preds.append(None)

        if (idx + 1) % 10 == 0:
            print(f"  已处理 {idx+1}/{len(df)} 条")

    df['Predicted_SideEffect'] = preds
    df.to_excel(output_file, index=False)
    print(f"预测结果已保存: {output_file}")


def main():
    if len(sys.argv) > 1:
        infile = sys.argv[1]
        outfile = sys.argv[2] if len(sys.argv) > 2 else None
        batch_mode(infile, outfile)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
