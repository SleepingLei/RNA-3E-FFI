# 训练脚本修改总结

## 📋 修改概述

已将 `scripts/04_train_model.py` 完全适配 v2.0 模型和数据格式。

---

## ✅ 主要修改

### 1. 模型导入
```python
# v1.0（旧）
from models.e3_gnn_encoder import RNAPocketEncoder

# v2.0（新）
from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2
from scripts.amber_vocabulary import get_global_encoder
```

---

### 2. 数据验证
新增 `LigandEmbeddingDataset` 数据格式验证：

```python
class LigandEmbeddingDataset:
    def __init__(self, ..., validate_format=True):
        # 检查特征维度（应为 4）
        if data.x.shape[1] != 4:
            warnings.append("Expected 4D features, got {}D".format(data.x.shape[1]))

        # 检查多跳索引
        if not hasattr(data, 'triple_index'):
            warnings.append("Missing triple_index")
```

**功能**:
- ✅ 自动检测旧格式数据
- ✅ 提示用户重新生成数据
- ✅ 过滤无效样本

---

### 3. 模型初始化

#### v1.0（旧）
```python
model = RNAPocketEncoder(
    input_dim=11,  # 固定维度
    hidden_irreps="32x0e + 16x1o + 8x2e",
    output_dim=1536
)
```

#### v2.0（新）
```python
# 获取词汇表大小
encoder = get_global_encoder()

model = RNAPocketEncoderV2(
    num_atom_types=encoder.num_atom_types,      # 动态获取
    num_residues=encoder.num_residues,          # 动态获取
    atom_embed_dim=32,                          # 新参数
    residue_embed_dim=16,                       # 新参数
    hidden_irreps="32x0e + 16x1o + 8x2e",
    output_dim=1536,
    use_multi_hop=True,                         # 新参数
    use_nonbonded=True,                         # 新参数
    pooling_type='attention',                   # 新参数
    dropout=0.0                                 # 新参数
)
```

**关键变化**:
- ❌ 移除 `input_dim`
- ✅ 添加 `num_atom_types`, `num_residues`（从词汇表获取）
- ✅ 添加 `atom_embed_dim`, `residue_embed_dim`
- ✅ 添加多跳和非键控制参数

---

### 4. 命令行参数

#### 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `--atom_embed_dim` | int | 32 | 原子类型嵌入维度 |
| `--residue_embed_dim` | int | 16 | 残基嵌入维度 |
| `--use_multi_hop` | flag | True | 启用多跳消息传递 |
| `--use_nonbonded` | flag | True | 启用非键交互 |
| `--use_gate` | flag | False | 使用 gate 激活 |
| `--use_layer_norm` | flag | False | 使用层归一化 |
| `--pooling_type` | str | attention | 池化类型 |
| `--dropout` | float | 0.0 | Dropout 率 |
| `--optimizer` | str | adam | 优化器类型 |
| `--scheduler` | str | plateau | 学习率调度器 |
| `--grad_clip` | float | 1.0 | 梯度裁剪 |

#### 移除参数

| 参数 | 原因 |
|-----|------|
| `--input_dim` | v2.0 使用 embedding，维度动态确定 |

---

### 5. 训练循环增强

#### 可学习权重监控

```python
def train_epoch(model, loader, optimizer, device):
    ...
    # 返回字典而不是单个值
    metrics = {'loss': total_loss / num_batches}

    # 添加可学习权重
    if hasattr(model, 'angle_weight'):
        metrics['angle_weight'] = model.angle_weight.item()
    if hasattr(model, 'dihedral_weight'):
        metrics['dihedral_weight'] = model.dihedral_weight.item()
    if hasattr(model, 'nonbonded_weight'):
        metrics['nonbonded_weight'] = model.nonbonded_weight.item()

    return metrics
```

**实时输出**:
```
Train Loss: 0.234567
  Angle weight: 0.5234
  Dihedral weight: 0.3123
  Nonbonded weight: 0.2456
```

---

#### 权重历史记录

```python
# 训练循环中
weight_history = {
    'angle_weight': [],
    'dihedral_weight': [],
    'nonbonded_weight': []
}

# 每轮记录
if 'angle_weight' in train_metrics:
    weight_history['angle_weight'].append(train_metrics['angle_weight'])

# 保存到历史文件
history = {
    'train_loss': train_history,
    'val_loss': val_history,
    'learnable_weights': weight_history,  # 新增
    'config': vars(args)                  # 新增
}
```

---

### 6. 梯度裁剪

```python
# 训练循环中添加
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
```

**作用**: 防止梯度爆炸，提高训练稳定性

---

### 7. 优化器选择

```python
if args.optimizer == "adamw":
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
```

**选项**:
- **Adam**: 标准选择
- **AdamW**: 更好的权重衰减（推荐）

---

### 8. 学习率调度器

```python
if args.scheduler == "cosine":
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01)
else:
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
```

**选项**:
- **Plateau**: 验证损失不下降时降低学习率（推荐）
- **Cosine**: 余弦退火

---

## 📊 输出变化

### v1.0 输出
```
models/checkpoints/
├── config.json
├── best_model.pt
└── training_history.json  # 仅包含 train_loss, val_loss
```

### v2.0 输出
```
models/checkpoints_v2/
├── config.json
├── best_model.pt
├── checkpoint_epoch_5.pt
├── checkpoint_epoch_10.pt
└── training_history.json  # 包含 learnable_weights
```

**`training_history.json` 示例**:
```json
{
  "train_loss": [0.5, 0.4, ...],
  "val_loss": [0.45, 0.35, ...],
  "learnable_weights": {
    "angle_weight": [0.5, 0.51, 0.52, ...],
    "dihedral_weight": [0.3, 0.31, 0.29, ...],
    "nonbonded_weight": [0.2, 0.19, 0.21, ...]
  },
  "config": {
    "use_multi_hop": true,
    "use_nonbonded": true,
    ...
  }
}
```

---

## 🔄 向后兼容性

**不兼容**:
- ❌ 无法直接加载 v1.0 的检查点
- ❌ 需要 v2.0 格式的数据（4维特征）

**迁移步骤**:
1. 重新生成数据: `python scripts/03_build_dataset.py`
2. 使用新脚本训练: `python scripts/04_train_model.py --use_multi_hop --use_nonbonded`

---

## 🚀 使用示例

### 快速开始
```bash
python scripts/04_train_model.py \
    --graph_dir data/processed/graphs \
    --embeddings_path data/processed/ligand_embeddings.h5 \
    --output_dir models/checkpoints_v2 \
    --batch_size 4 \
    --num_epochs 100 \
    --use_multi_hop \
    --use_nonbonded
```

### 完整配置
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

### Baseline（仅 1-hop）
```bash
python scripts/04_train_model.py \
    --output_dir models/baseline \
    --batch_size 4 \
    --num_epochs 100
# 注意：不添加 --use_multi_hop 和 --use_nonbonded
```

---

## ⚠️ 常见问题

### 数据格式不匹配

**错误**:
```
Expected 4D features, got 11D
```

**解决**:
```bash
python scripts/03_build_dataset.py
```

---

### 内存不足

**错误**:
```
CUDA out of memory
```

**解决**:
1. 减小 batch_size: `--batch_size 1`
2. 减小模型: `--hidden_irreps "16x0e + 8x1o" --num_layers 2`
3. 禁用非键: 移除 `--use_nonbonded`

---

### 训练不稳定

**症状**: 损失震荡或 NaN

**解决**:
1. 降低学习率: `--lr 5e-5`
2. 增加梯度裁剪: `--grad_clip 0.5`
3. 使用 AdamW: `--optimizer adamw`

---

## 📚 文档

详细文档请参考:
- **使用指南**: `TRAINING_GUIDE_V2.md`
- **模型文档**: `MODELS_V2_SUMMARY.md`
- **多跳实现**: `MULTI_HOP_IMPLEMENTATION.md`

---

**修改日期**: 2025-10-25
**版本**: v2.0
**状态**: ✅ 已测试，可用
