# RNA-3E-FFI v2.0 训练指南

## 📋 概述

本指南介绍如何使用修改后的训练脚本 `scripts/04_train_model.py` 训练 v2.0 模型。

---

## 🚀 快速开始

### 1. 准备数据（v2.0 格式）

确保已经使用 v2.0 格式重新生成数据：

```bash
# 重新生成图数据（4维特征）
python scripts/03_build_dataset.py \
    --hariboss_csv hariboss/Complexes.csv \
    --amber_dir data/processed/amber \
    --output_dir data/processed/graphs \
    --distance_cutoff 5.0

# 检查数据格式
python test_v2_features.py
```

**必需文件**:
- `data/processed/graphs/*.pt` - 图数据（v2.0 格式）
- `data/processed/ligand_embeddings.h5` - 配体嵌入
- `hariboss/Complexes.csv` - 复合物列表

---

### 2. 基础训练（推荐配置）

```bash
python scripts/04_train_model.py \
    --graph_dir data/processed/graphs \
    --embeddings_path data/processed/ligand_embeddings.h5 \
    --output_dir models/checkpoints_v2 \
    --batch_size 4 \
    --num_epochs 100 \
    --lr 1e-4 \
    --use_multi_hop \
    --use_nonbonded \
    --pooling_type attention
```

**说明**:
- `--use_multi_hop`: 启用 2/3-hop 消息传递
- `--use_nonbonded`: 启用非键交互
- `--pooling_type attention`: 使用注意力池化

---

### 3. 高级配置

#### 完整的多跳 + 非键模型

```bash
python scripts/04_train_model.py \
    --graph_dir data/processed/graphs \
    --embeddings_path data/processed/ligand_embeddings.h5 \
    --output_dir models/checkpoints_full \
    --batch_size 2 \
    --num_epochs 300 \
    --lr 1e-4 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --use_multi_hop \
    --use_nonbonded \
    --hidden_irreps "64x0e + 32x1o + 16x2e" \
    --num_layers 5 \
    --dropout 0.1 \
    --pooling_type attention
```

#### 仅 1-hop（Baseline）

```bash
python scripts/04_train_model.py \
    --output_dir models/checkpoints_baseline \
    --batch_size 4 \
    --num_epochs 100 \
    --hidden_irreps "32x0e + 16x1o + 8x2e" \
    --num_layers 3 \
    # 注意：不添加 --use_multi_hop 和 --use_nonbonded
```

#### 小规模测试

```bash
python scripts/04_train_model.py \
    --output_dir models/test \
    --batch_size 8 \
    --num_epochs 10 \
    --num_layers 2 \
    --hidden_irreps "16x0e + 8x1o" \
    --use_multi_hop \
    --train_ratio 0.1 \
    --val_ratio 0.05
```

---

## 🎛️ 命令行参数详解

### 数据参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--hariboss_csv` | `hariboss/Complexes.csv` | HARIBOSS CSV 文件 |
| `--graph_dir` | `data/processed/graphs` | 图数据目录 |
| `--embeddings_path` | `data/processed/ligand_embeddings.h5` | 配体嵌入文件 |
| `--splits_file` | `data/splits/splits.json` | 数据划分文件 |

---

### 模型参数（v2.0 新增）

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--atom_embed_dim` | 32 | 原子类型嵌入维度 |
| `--residue_embed_dim` | 16 | 残基嵌入维度 |
| `--hidden_irreps` | `32x0e + 16x1o + 8x2e` | 隐藏层 irreps |
| `--output_dim` | 1536 | 输出嵌入维度 |
| `--num_layers` | 4 | 消息传递层数 |
| `--num_radial_basis` | 8 | 径向基函数数量 |

**Irreps 配置指南**:
```
"32x0e + 16x1o + 8x2e"
 ↓      ↓      ↓
标量   向量   二阶张量

- 32x0e: 32 个标量特征（不变量）
- 16x1o: 16 个向量特征（等变）
- 8x2e: 8 个二阶张量特征（等变）
```

**推荐配置**:
- **小模型**: `"16x0e + 8x1o"`
- **中等模型**: `"32x0e + 16x1o + 8x2e"` (默认)
- **大模型**: `"64x0e + 32x1o + 16x2e + 8x3o"`

---

### v2.0 专属参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--use_multi_hop` | `True` | 启用多跳消息传递 |
| `--use_nonbonded` | `True` | 启用非键交互 |
| `--use_gate` | `False` | 使用 gate 激活（需要 improved layers）|
| `--use_layer_norm` | `False` | 使用层归一化（需要 improved layers）|
| `--pooling_type` | `attention` | 池化类型 (`attention`, `mean`, `sum`, `max`) |
| `--dropout` | 0.0 | Dropout 率 |

**多跳配置说明**:
- `--use_multi_hop`: 包含 2-hop 角度 + 3-hop 二面角
- `--use_nonbonded`: 使用 LJ 参数的非键边
- 同时启用两者可获得最佳性能

---

### 训练参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--batch_size` | 2 | 批大小 |
| `--num_epochs` | 300 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--weight_decay` | 1e-5 | 权重衰减 |
| `--optimizer` | `adam` | 优化器 (`adam`, `adamw`) |
| `--scheduler` | `plateau` | 学习率调度器 (`plateau`, `cosine`) |
| `--patience` | 10 | 早停耐心值 |
| `--grad_clip` | 1.0 | 梯度裁剪（0 禁用）|
| `--num_workers` | 4 | 数据加载器线程数 |

**优化器选择**:
- **Adam**: 标准选择，稳定
- **AdamW**: 更好的正则化，推荐用于大模型

**调度器选择**:
- **plateau**: 验证损失不下降时降低学习率（推荐）
- **cosine**: 余弦退火，适合长时间训练

---

### 数据划分参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--train_ratio` | 0.8 | 训练集比例 |
| `--val_ratio` | 0.1 | 验证集比例 |
| `--seed` | 42 | 随机种子 |

---

### 输出参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--output_dir` | `models/checkpoints` | 检查点保存目录 |
| `--save_every` | 5 | 每 N 轮保存一次 |
| `--resume` | False | 从检查点恢复训练 |
| `--checkpoint` | `models/checkpoints/best_model.pt` | 检查点路径 |

---

## 📊 训练输出

### 实时监控

训练过程中会显示：

```
Epoch 10/100
------------------------------------------------------------
Training: 100%|██████████| 50/50 [01:23<00:00,  1.67s/it]
Train Loss: 0.234567
  Angle weight: 0.5234        # 可学习权重（实时更新）
  Dihedral weight: 0.3123
  Nonbonded weight: 0.2456
Evaluating: 100%|██████████| 10/10 [00:15<00:00,  1.52s/it]
Val Loss: 0.198765, Val L1: 0.123456
Learning Rate: 1.00e-04
New best model! Saved to models/checkpoints/best_model.pt
```

**关键指标**:
- **Train Loss**: 训练集 MSE 损失
- **Val Loss**: 验证集 MSE 损失
- **Val L1**: 验证集 L1 损失
- **Angle/Dihedral/Nonbonded weight**: 可学习组合权重

---

### 保存的文件

训练完成后会生成：

```
models/checkpoints_v2/
├── config.json                    # 训练配置
├── training_history.json          # 训练历史（包含可学习权重）
├── best_model.pt                  # 最佳模型
├── checkpoint_epoch_5.pt          # 周期性检查点
├── checkpoint_epoch_10.pt
└── ...
```

**`training_history.json` 格式**:
```json
{
  "train_loss": [0.5, 0.4, 0.3, ...],
  "val_loss": [0.45, 0.35, 0.28, ...],
  "learnable_weights": {
    "angle_weight": [0.5, 0.51, 0.52, ...],
    "dihedral_weight": [0.3, 0.31, 0.29, ...],
    "nonbonded_weight": [0.2, 0.19, 0.21, ...]
  },
  "config": {...}
}
```

---

## 🔧 常见问题

### 1. 数据格式错误

**错误信息**:
```
⚠️  Format validation warnings (15 total):
  - 1ei2_NMY: Expected 4D features, got 11D
```

**解决方案**:
```bash
# 重新生成 v2.0 格式的图数据
python scripts/03_build_dataset.py
```

---

### 2. 内存不足（OOM）

**症状**: CUDA out of memory

**解决方案**:
1. 减小 batch size:
   ```bash
   --batch_size 1
   ```

2. 减少模型大小:
   ```bash
   --hidden_irreps "16x0e + 8x1o" \
   --num_layers 2
   ```

3. 禁用非键交互（内存占用大）:
   ```bash
   # 不添加 --use_nonbonded
   ```

4. 减少 num_workers:
   ```bash
   --num_workers 0
   ```

---

### 3. 训练不收敛

**症状**: 损失不下降或震荡

**解决方案**:

1. 降低学习率:
   ```bash
   --lr 5e-5
   ```

2. 增加梯度裁剪:
   ```bash
   --grad_clip 0.5
   ```

3. 使用更稳定的优化器:
   ```bash
   --optimizer adamw \
   --weight_decay 1e-4
   ```

4. 添加 dropout:
   ```bash
   --dropout 0.1
   ```

---

### 4. 多跳路径缺失警告

**警告信息**:
```
Missing triple_index (2-hop angles)
```

**影响**: 多跳功能将不可用

**解决方案**:
- 确保使用最新的 `scripts/03_build_dataset.py` 生成数据
- 或禁用多跳: 不添加 `--use_multi_hop`

---

## 📈 消融实验建议

### 实验 1: Baseline（1-hop only）

```bash
python scripts/04_train_model.py \
    --output_dir models/ablation/baseline \
    --batch_size 4 \
    --num_epochs 100
```

### 实验 2: + Multi-hop

```bash
python scripts/04_train_model.py \
    --output_dir models/ablation/multi_hop \
    --batch_size 4 \
    --num_epochs 100 \
    --use_multi_hop
```

### 实验 3: + Non-bonded

```bash
python scripts/04_train_model.py \
    --output_dir models/ablation/nonbonded \
    --batch_size 4 \
    --num_epochs 100 \
    --use_multi_hop \
    --use_nonbonded
```

### 实验 4: Full model

```bash
python scripts/04_train_model.py \
    --output_dir models/ablation/full \
    --batch_size 2 \
    --num_epochs 100 \
    --use_multi_hop \
    --use_nonbonded \
    --dropout 0.1
```

---

## 🔄 恢复训练

从检查点恢复：

```bash
python scripts/04_train_model.py \
    --resume \
    --checkpoint models/checkpoints_v2/best_model.pt \
    --output_dir models/checkpoints_v2 \
    --num_epochs 500  # 继续训练到 500 轮
```

**注意**:
- 确保使用相同的模型配置
- 训练历史会自动加载

---

## 📊 监控可学习权重

训练结束后，检查权重演化：

```python
import json

# 加载训练历史
with open('models/checkpoints_v2/training_history.json', 'r') as f:
    history = json.load(f)

# 可学习权重
weights = history['learnable_weights']
print(f"Final angle weight: {weights['angle_weight'][-1]:.4f}")
print(f"Final dihedral weight: {weights['dihedral_weight'][-1]:.4f}")
print(f"Final nonbonded weight: {weights['nonbonded_weight'][-1]:.4f}")
```

**示例输出**:
```
Final angle weight: 0.5234 (initial: 0.500)
Final dihedral weight: 0.2891 (initial: 0.300)
Final nonbonded weight: 0.2156 (initial: 0.200)
```

**解释**:
- 权重变化表明模型自动学习到了最优组合
- 角度权重增加 → 角度信息更重要
- 二面角权重减小 → 二面角贡献较小

---

## 💡 最佳实践

### 1. 数据准备
- ✅ 使用 v2.0 格式（4维特征）
- ✅ 确保包含多跳索引
- ✅ 检查 LJ 参数是否提取成功

### 2. 超参数选择
- **小规模测试**: batch_size=8, num_layers=2, num_epochs=10
- **完整训练**: batch_size=2-4, num_layers=4-5, num_epochs=100-300
- **学习率**: 从 1e-4 开始，使用 plateau 调度器

### 3. 模型配置
- **推荐**: `use_multi_hop=True`, `use_nonbonded=True`
- **池化**: attention > mean > sum
- **优化器**: AdamW > Adam

### 4. 监控指标
- 主要关注 **Val Loss**
- 监控 **可学习权重** 的演化
- 检查是否过拟合（train loss << val loss）

---

## 🎓 下一步

训练完成后：

1. **评估模型**:
   ```bash
   python scripts/05_run_inference.py \
       --model_path models/checkpoints_v2/best_model.pt \
       --test_split data/splits/splits.json
   ```

2. **可视化训练曲线**:
   ```python
   import matplotlib.pyplot as plt
   import json

   with open('models/checkpoints_v2/training_history.json') as f:
       history = json.load(f)

   plt.plot(history['train_loss'], label='Train')
   plt.plot(history['val_loss'], label='Val')
   plt.legend()
   plt.savefig('training_curves.png')
   ```

3. **分析可学习权重**:
   - 查看权重如何随训练演化
   - 理解不同相互作用的相对重要性

---

**版本**: v2.0
**更新**: 2025-10-25
**状态**: ✅ 已测试，可用
