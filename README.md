# Side_effect_code
# 药物-菌群-副作用预测系统

## 系统概述

本系统是一个端到端的药物副作用预测平台，通过整合药物分子结构、菌群蛋白序列、药物-蛋白结合亲和力以及菌群丰度变化方向，预测药物可能引发的副作用类别。


###1 训练流程

**脚本**: `train_side_effect_model.py`

#### 数据准备

| 步骤 | 说明 |
|------|------|
| 读取数据 | `side_with_smiles.xlsx` |
| 类别筛选 | 保留样本数前 8 的副作用类别 |
| 标签编码 | `LabelEncoder` 将类别映射为 0-7 |
| 特征提取 | Morgan 指纹 + 蛋白特征 + 结合分数 + 变化方向 |

#### 2 特征工程

| 特征类型 | 维度 | 说明 |
|---------|------|------|
| Morgan 指纹 (ECFP) | 2048 | 半径 2，2048 位，捕获分子子结构 |
| 氨基酸组成 | 20 | 20 种氨基酸在蛋白中的比例 |
| 蛋白统计特征 | 5 | 相对长度、疏水性、带电性、极性、半胱氨酸比例 |
| 结合预测分数 | 1 | 结合模型预测的亲和力值 |
| 菌种变化方向 | 1 | increase=1, decrease=0 |
| **总计** | **2075** | |

#### 3 模型对比

训练 9 种模型（8 传统 + 1 深度学习），5 折交叉验证：

| 模型 | 说明 |
|------|------|
| XGBoost | 梯度提升树 |
| LightGBM | 快速梯度提升 |
| RandomForest | 随机森林 |
| ExtraTrees | 极端随机树 |
| AdaBoost | 自适应提升 |
| GradientBoosting | 经典梯度提升 |
| LogisticRegression | 逻辑回归（基线） |
| SVM | 支持向量机（RBF 核） |
| **DeepLearning (DNN)** | **4 层全连接网络** |

###  深度学习模型结构

**类名**: `SideEffectNet`

```
输入 (2075 维)
    │
    ▼
┌─────────────────────┐
│ Linear(2075, 512)   │
│ BatchNorm1d(512)    │
│ GELU                │
│ Dropout(0.4)        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Linear(512, 256)    │
│ BatchNorm1d(256)    │
│ GELU                │
│ Dropout(0.4)        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Linear(256, 128)    │
│ BatchNorm1d(128)    │
│ GELU                │
│ Dropout(0.4)        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Linear(128, 64)     │
│ BatchNorm1d(64)     │
│ GELU                │
│ Dropout(0.4)        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Linear(64, 8)       │
│ (8 个副作用类别)     │
└─────────────────────┘
```

**训练策略**：
- 损失函数：交叉熵 + 标签平滑（smoothing=0.1）
- 优化器：AdamW（weight_decay=0.01）
- 学习率调度：10 epoch 预热 + CosineAnnealing
- 梯度裁剪：max_norm=1.0
- 数据增强：高斯噪声（σ=0.01）
- 早停：30 轮无提升自动停止，最大 200 epoch

### 模型文件

| 文件 | 说明 |
|------|------|
| `side_effect_models/best_deep_model.pth` | 深度学习模型权重 |
| `side_effect_models/label_encoder.pkl` | 标签编码器（类别↔数字映射） |
| `side_effect_models/scaler.pkl` | 特征标准化器 |
| `side_effect_models/model_*.pkl` | 各传统机器学习模型 |

---

## 四、预测系统

###  交互模式

**脚本**: `predict_side_effect.py`

```bash
python3 predict_side_effect.py
```

**输入**：
1. 药物 SMILES 结构
2. 菌群蛋白序列
3. 菌群变化方向（increase/decrease）

**输出**：
- 预测副作用类别
- 置信度
- Top 3 预测结果
- 因果链分析

###  批量模式

```bash
python3 predict_side_effect.py input.xlsx output.xlsx
```

Excel 需包含列：`SMILES`、`Protein_Sequence`、`变化方向`

###  因果链分析

使用**扰动分析法**追踪预测决策路径：

1. **基准预测**：正常输入得到预测概率
2. **特征扰动**：将某一组特征置零，重新预测
3. **贡献度计算**：概率下降幅度 = 该特征的贡献度
4. **归一化**：各特征贡献度归一化为百分比

**输出内容**：
- 完整因果链可视化
- 各特征组贡献度排名
- 所有类别概率分布
- 药物活性子结构数量
- 蛋白 Top 5 氨基酸组成

---

## 输出数据说明

###  最终 Excel 文件列说明

| 列名 | 类型 | 说明 |
|------|------|------|
| `药物名称` | 字符串 | 原始药物名称（可能为中文） |
| `SMILES` | 字符串 | 药物分子 SMILES 结构式 |
| `第一个菌种` | 字符串 | 原始菌种名称 |
| `Protein_Accession` | 字符串 | NCBI 蛋白序列登录号 |
| `Protein_Sequence` | 字符串 | 菌种蛋白氨基酸序列 |
| `Predicted_Affinity` | 浮点数 | 药物-蛋白结合亲和力预测值 |
| `变化方向` | 字符串 | increase（上升）或 decrease（下降） |
| `副作用分类(英文)` | 字符串 | 副作用类别标签 |
| `Predicted_SideEffect` | 字符串 | 模型预测的副作用类别 |

###  副作用类别说明

系统保留样本数前 8 的类别（具体类别取决于训练数据分布），常见类别包括：

| 类别 | 说明 |
|------|------|
| Gastrointestinal | 胃肠道反应（恶心、呕吐、腹泻等） |
| Metabolic/Endocrine | 代谢/内分泌异常 |
| Liver | 肝毒性 |
| Neurological | 神经系统反应 |
| Hematological | 血液系统反应 |
| Renal | 肾毒性 |
| Immune-related | 免疫相关反应 |
| Respiratory | 呼吸系统反应 |

###  可视化输出

训练完成后自动生成：

| 文件 | 说明 |
|------|------|
| `model_comparison.png` | 各模型准确率对比柱状图 |
| `class_distribution.png` | 副作用类别分布（饼图+条形图） |
| `per_class_metrics.png` | 各类别 Precision/Recall/F1 对比 |
| `prediction_summary.xlsx` | 各类别预测性能汇总表 |

---

## 环境依赖

```bash
# 基础依赖
pip install pandas numpy openpyxl requests openai

# 化学信息学
pip install rdkit pubchempy

# 深度学习
pip install torch torch-geometric

# 机器学习
pip install scikit-learn xgboost lightgbm joblib

# 可视化
pip install matplotlib seaborn
```

---

## 完整运行流程

```bash

#  训练副作用预测模型
python3 train_side_effect_model.py

#  使用模型进行预测（交互模式）
python3 predict_side_effect.py

#  使用模型进行预测（批量模式）
python3 predict_side_effect.py input.xlsx output.xlsx
```

