# 输入特征监控功能 - 完成总结

## ✅ 已完成功能

梯度爆炸诊断脚本现已集成完整的输入特征监控功能，可以实时追踪所有输入数据的统计信息和异常变化。

---

## 🎯 监控的特征

### 1. 节点特征 (x)
- **x_col0**: 电荷 (charge) - 归一化后的原子电荷
- **x_col1**: 原子序数 (atomic_num) - 原子的原子序数
- **x_col2**: 质量 (mass) - 原子质量

**监控指标**: mean, std, max, min, NaN/Inf检测

### 2. 位置 (pos)
- 3D坐标 (x, y, z)

**监控指标**: mean, std, max, min, NaN/Inf检测

### 3. 边特征 (edge_attr)
- **col0**: req/2.0 - 归一化的平衡键长
- **col1**: k/500.0 - 归一化的键力常数

**监控指标**: mean, std, max, min, NaN检测

### 4. 角度特征 (triple_attr)
- **col0**: theta_eq/180.0 - 归一化的平衡角度
- **col1**: k/200.0 - 归一化的角度力常数

**监控指标**: mean, std, max, min

### 5. 二面角特征 (quadra_attr)
- **col0**: phi_k/20.0 - 归一化的二面角力常数
- **col1**: per/6.0 - 归一化的周期数
- **col2**: phase/(2π) - 归一化的相位

**监控指标**: mean, std, max, min

---

## 🔍 异常检测机制

### 1. NaN/Inf 检测
自动检测所有特征中的NaN和Inf值，立即报警。

### 2. 超出预期范围检测
每个特征都有预定义的合理范围：
```python
expected_ranges = {
    'x_col0': (-2, 2),       # charge
    'x_col1': (0, 100),      # atomic_num
    'x_col2': (0, 300),      # mass
    'pos': (-100, 100),      # positions
    'edge_attr_col0': (0, 2),      # req
    'edge_attr_col1': (0, 1),      # k
    'triple_attr_col0': (0, 2),    # theta_eq
    'triple_attr_col1': (0, 1),    # k
    'quadra_attr_col0': (0, 1),    # phi_k
    'quadra_attr_col1': (0, 1),    # per
    'quadra_attr_col2': (0, 1),    # phase
}
```

如果特征值超出范围，会被标记为异常。

### 3. 突然变化检测
使用**滑动窗口 + z-score**方法检测突变：
- 窗口大小: 10 步
- 阈值: z-score > 2.0
- 计算最近10步的均值和标准差，与当前值比较

如果当前值偏离历史均值超过2个标准差，则认为发生突变。

---

## 📊 新增输出文件

### 1. `input_features.csv`

**位置**: `{output_dir}/input_features.csv`

**内容**: 每个训练step的所有输入特征统计

**列名**:
```
step, epoch, batch_idx,
x_col0_mean, x_col0_std, x_col0_max, x_col0_min,
x_col1_mean, x_col1_std, x_col1_max, x_col1_min,
x_col2_mean, x_col2_std, x_col2_max, x_col2_min,
pos_mean, pos_std, pos_max, pos_min,
edge_attr_col0_mean, edge_attr_col0_std,
edge_attr_col1_mean, edge_attr_col1_std,
triple_attr_col0_mean, triple_attr_col0_std,
triple_attr_col1_mean, triple_attr_col1_std,
suspicious_features
```

**用途**:
- 导入Excel/Pandas进行详细分析
- 识别哪些特征在训练过程中发生异常变化
- 关联特征变化与梯度爆炸的时间点

### 2. `feature_changes.png`

**位置**: `{output_dir}/feature_changes.png`

**内容**: 3x3子图网格，显示9个关键特征的时间序列

**子图列表**:
1. Node Feature: Charge (mean)
2. Node Feature: Atomic Number (mean)
3. Node Feature: Mass (mean)
4. Position (mean)
5. Position (std)
6. Edge: r_eq (mean)
7. Edge: k (mean)
8. Angle: theta_eq (mean)
9. Angle: k (mean)

**特点**:
- 自动标注异常值（红色散点）
- 使用3σ准则检测异常
- 网格布局便于对比

**查看方式**:
```bash
open gradient_diagnosis/feature_changes.png
```

---

## 📝 JSON报告更新

`diagnosis_report.json` 现在包含 `input_feature_stats` 字段：

```json
{
  "input_feature_stats": {
    "latest": {
      "step": 1000,
      "x_col0_mean": 0.123,
      "x_col1_mean": 12.45,
      ...
    },
    "suspicious_features": [
      {
        "feature": "x_col0",
        "issue": "Max value 3.5 exceeds expected 2.0",
        "step": 850
      }
    ],
    "sudden_changes": [
      {
        "feature": "pos_mean",
        "z_score": 3.5,
        "prev_mean": 12.3,
        "latest_value": 18.7,
        "change": 6.4
      }
    ]
  }
}
```

---

## 🚀 如何使用

### 运行诊断脚本
```bash
python scripts/diagnose_gradient_explosion.py \
    --model_version v3 \
    --data_dir data/processed_pockets \
    --ligand_embeddings data/ligand_embeddings.h5 \
    --split_dir data/splits \
    --epochs 10 \
    --output_dir gradient_diagnosis
```

### 查看特征变化
```bash
# 查看可视化
open gradient_diagnosis/feature_changes.png

# 分析CSV数据
python -c "
import pandas as pd
df = pd.read_csv('gradient_diagnosis/input_features.csv')

# 查看特征异常
anomalies = df[df['suspicious_features'] != '']
print('特征异常发生在以下steps:')
print(anomalies[['step', 'epoch', 'suspicious_features']])

# 统计特征范围
print(f'\nCharge range: [{df[\"x_col0_min\"].min():.2f}, {df[\"x_col0_max\"].max():.2f}]')
print(f'Position std range: [{df[\"pos_std\"].min():.2f}, {df[\"pos_std\"].max():.2f}]')
"
```

### 实时监控输出

训练过程中会显示特征异常：
```
⚠️  Step 145 - 检测到异常!
  梯度异常层: [...]
  输入特征异常: [
    {'feature': 'x_col0', 'issue': 'Max value 3.2 exceeds expected 2.0', 'step': 145}
  ]
  特征突变: [
    {'feature': 'pos_mean', 'z_score': 3.2, 'change': 5.6}
  ]
  Total grad norm: 45.6789
```

---

## 💡 典型应用场景

### 场景 1: 梯度爆炸前特征突变

**症状**: 梯度在step 500爆炸

**分析**:
```python
df = pd.read_csv('gradient_diagnosis/input_features.csv')

# 查看step 480-500的特征变化
recent = df[(df['step'] >= 480) & (df['step'] <= 500)]
print(recent[['step', 'x_col0_mean', 'pos_mean', 'suspicious_features']])
```

**可能发现**:
- step 495: `pos_mean` 从12.3突增至18.7
- step 498: 梯度开始增大
- step 500: 梯度爆炸

**结论**: 某个样本的位置数据异常，需要排查数据加载器

---

### 场景 2: 数据归一化问题

**症状**: 训练初期就不稳定

**分析**:
```python
df = pd.read_csv('gradient_diagnosis/input_features.csv')

# 查看前20步的特征范围
early = df[df['step'] <= 20]
print("x_col0 range:", early['x_col0_min'].min(), "~", early['x_col0_max'].max())
print("Expected range: -2 ~ 2")
```

**可能发现**:
- `x_col0_max` = 5.6 (超出预期的 2.0)
- 数据没有正确归一化

**结论**: 需要修复数据预处理脚本

---

### 场景 3: 特定样本异常

**症状**: 某个epoch突然爆炸

**分析**:
```python
df = pd.read_csv('gradient_diagnosis/input_features.csv')
anomalies = df[df['suspicious_features'] != '']

# 找出异常样本对应的step
print(anomalies[['step', 'epoch', 'batch_idx', 'suspicious_features']])
```

**可能发现**:
- epoch 4, batch 23: `edge_attr_col0` 出现NaN

**结论**: 数据集中第23个batch包含错误数据，需要清洗

---

## 📚 相关文档

已更新的文档：
- **`docs/gradient_explosion_diagnosis_guide.md`** - 完整使用指南
- **`GRADIENT_DIAGNOSIS_QUICKSTART.md`** - 快速开始指南

新增内容：
- 输入特征监控说明
- `input_features.csv` 使用方法
- `feature_changes.png` 解读
- 特征异常诊断流程

---

## 🎉 总结

现在梯度爆炸诊断脚本具备了**全方位**的监控能力：

✅ **梯度监控** - 每层梯度统计
✅ **激活值监控** - 每层激活值分布
✅ **权重监控** - 权重变化追踪
✅ **输入特征监控** - 所有输入数据统计 (NEW)
✅ **实时异常检测** - NaN/Inf/超范围/突变
✅ **自动可视化** - 4个图表全面展示
✅ **详细CSV日志** - 便于后续分析

使用这个工具可以：
1. 快速定位梯度爆炸的根本原因
2. 区分是模型问题还是数据问题
3. 找出有问题的数据样本
4. 验证数据预处理是否正确
5. 追踪特征变化与梯度的关联

**立即试用**:
```bash
python scripts/diagnose_gradient_explosion.py \
    --model_version v3 \
    --data_dir data/processed_pockets \
    --ligand_embeddings data/ligand_embeddings.h5 \
    --split_dir data/splits \
    --epochs 10 \
    --output_dir gradient_diagnosis

# 查看结果
open gradient_diagnosis/feature_changes.png
```

---

*版本: 1.0*
*创建时间: 2025-11-10*
*功能: 输入特征监控*
