# NaN问题修复指南

## 🔍 问题回顾

**症状**: 无论使用什么loss函数或网络层数，只要不使用LayerNorm就会出现NaN

**根本原因**: 使用`@property`装饰器实现权重约束时，与PyTorch的Autograd机制不兼容

---

## ✅ 已完成的修复

### 1. 修改模型定义 (`models/e3_gnn_encoder_v2.py`)

**Before**:
```python
@property
def angle_weight(self):
    if hasattr(self, 'angle_weight_raw'):
        return torch.exp(torch.clamp(self.angle_weight_raw, min=-5, max=5))
    return None  # ← 问题：可能返回None
```

**After**:
```python
def get_angle_weight(self):
    """Get angle weight (ensures it stays positive and bounded)."""
    return torch.exp(torch.clamp(self.angle_weight_raw, min=-5, max=5))
```

### 2. 更新forward方法
```python
# Before
h_new = h_new + self.angle_weight * h_angle

# After
h_new = h_new + self.get_angle_weight() * h_angle
```

### 3. 更新训练脚本 (`scripts/04_train_model.py`)
```python
# Before
model.angle_weight.item()

# After
model.get_angle_weight().item()
```

---

## 🧪 测试修复

### 步骤1: 快速单元测试（在远程运行）

```bash
chmod +x test_weight_fix.sh
bash test_weight_fix.sh
```

**预期输出**:
```
✓ angle_weight正常
✓ angle_weight_raw梯度正常
✓ 所有检查通过！
```

### 步骤2: 完整训练测试（3个epoch）

```bash
chmod +x test_nan_fix.sh
bash test_nan_fix.sh
```

**预期输出**:
```
Batch 0: Grad norm before clip = 20-50 (不是NaN)
Train Loss: 0.95, Cosine Sim: 0.05 (不是NaN)
Angle weight: 0.333 (不是NaN)
```

**如果仍然NaN**: 运行详细诊断
```bash
python scripts/debug_exact_nan_location.py
```

---

## 🚀 正式训练

### 推荐配置（修复后）

```bash
python scripts/04_train_model.py \
    --embeddings_path data/processed/ligand_embeddings.h5 \
    --output_dim 1536 \
    --batch_size 4 \
    --num_epochs 300 \
    --lr 5e-4 \
    --num_workers 1 \
    --use_multi_hop \
    --use_nonbonded \
    --use_gate \
    --save_every 5 \
    --num_layers 6 \
    --dropout 0.1 \
    --loss_fn cosine \
    --monitor_gradients \
    --output_dir models/checkpoints_cosine_fixed
```

**关键参数**:
- `--lr 5e-4` - 较小的学习率（从2e-3降低）
- `--batch_size 4` - 增大batch size（从2增加）
- `--loss_fn cosine` - 使用余弦相似度loss
- `--monitor_gradients` - 监控梯度范数

---

## 📊 预期训练表现

### 健康的训练应该看到：

```
Epoch 1/300
------------------------------------------------------------
  Batch 0: Grad norm before clip = 25.34
  Batch 50: Grad norm before clip = 12.56
  Batch 100: Grad norm before clip = 8.91
Train Loss: 0.95, Cosine Sim: 0.05
  Angle weight: 0.333
  Dihedral weight: 0.333
  Nonbonded weight: 0.333
Val Loss: 0.94, Cosine Sim: 0.06

Epoch 2/300
------------------------------------------------------------
Train Loss: 0.89, Cosine Sim: 0.11  ← Loss下降
Val Loss: 0.88, Cosine Sim: 0.12

Epoch 10/300
------------------------------------------------------------
Train Loss: 0.65, Cosine Sim: 0.35  ← 持续改善
Val Loss: 0.67, Cosine Sim: 0.33
```

### 关键指标：

- ✅ **梯度范数**: 5-50之间（不是NaN，不是0.1）
- ✅ **Train Loss**: 逐渐下降（从0.95 → 0.65）
- ✅ **Cosine Sim**: 逐渐上升（从0.05 → 0.35）
- ✅ **权重值**: 稳定在0.1-2.0范围（不是NaN）

---

## ⚠️ 如果还有问题

### 问题1: 仍然出现NaN

运行详细诊断：
```bash
python scripts/debug_exact_nan_location.py
```

这会告诉您NaN出现在哪个具体位置。

### 问题2: 梯度仍然很小（< 1.0）

可能是LayerNorm的问题。尝试：
```bash
# 使用LayerNorm（更稳定，但梯度小）
--use_layer_norm --num_layers 4

# 或减少层数
--num_layers 3
```

### 问题3: Loss不下降

检查：
1. 学习率是否太小？尝试 `--lr 1e-3`
2. Batch size是否太小？尝试 `--batch_size 8`
3. 数据是否正确？运行 `python scripts/debug_exact_nan_location.py`

---

## 📁 相关文档

- `docs/PROPERTY_FIX.md` - Property vs 普通方法详细对比
- `docs/NAN_ISSUE_ROOT_CAUSE.md` - NaN问题的完整分析
- `docs/COSINE_LOSS_TROUBLESHOOTING.md` - Cosine loss特定问题

---

## 🎯 总结

**修复内容**:
1. ✅ 将`@property`改为普通方法`get_weight()`
2. ✅ 更新forward和训练脚本中的所有调用
3. ✅ 输出层保留LayerNorm作为数值稳定性保护

**预期效果**:
- ✅ 不再出现NaN
- ✅ 梯度范数正常（5-50）
- ✅ Loss能正常下降
- ✅ 权重能正常学习

**立即测试**:
```bash
bash test_nan_fix.sh
```

如果看到正常的Loss值（不是NaN），说明修复成功！🎉
