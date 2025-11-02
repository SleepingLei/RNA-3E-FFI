# Cosine Loss更新说明

**日期**: 2025-11-02
**修改文件**: `scripts/04_train_model.py`
**目的**: 将损失函数从MSE替换为Cosine Similarity Loss

---

## ✅ 已完成的修改

### 1. 训练函数 (train)

**修改位置**: 第179-182行

**修改前**:
```python
# MSE loss
loss = F.mse_loss(pocket_embedding, target_embedding)
```

**修改后**:
```python
# Cosine Similarity Loss (1 - cosine_similarity)
# We want to maximize cosine similarity, so minimize (1 - cosine_similarity)
cosine_sim = F.cosine_similarity(pocket_embedding, target_embedding, dim=1)
loss = (1 - cosine_sim).mean()
```

---

### 2. 评估函数 (evaluate)

**修改位置**: 第232-275行

**主要变化**:
- **主要指标**: 从MSE改为Cosine Loss
- **新增指标**: 平均余弦相似度 (avg_cosine_similarity)
- **保留指标**: MSE (仅用于对比)

**返回值**:
```python
return {
    'cosine_loss': ...,          # 主要损失 (新)
    'avg_cosine_similarity': ..., # 新增指标
    'mse_loss': ...              # 保留用于对比
}
```

---

### 3. 验证日志输出

**修改位置**: 第665-668行

**修改前**:
```python
val_loss = val_metrics['mse_loss']
print(f"Val Loss: {val_loss:.6f}, Val L1: {val_metrics['l1_loss']:.6f}")
```

**修改后**:
```python
val_loss = val_metrics['cosine_loss']
val_cosine_sim = val_metrics['avg_cosine_similarity']
val_mse = val_metrics['mse_loss']
print(f"Val Cosine Loss: {val_loss:.6f}, Val Cosine Sim: {val_cosine_sim:.4f}, Val MSE: {val_mse:.4f}")
```

**新的输出示例**:
```
Val Cosine Loss: 0.2341, Val Cosine Sim: 0.7659, Val MSE: 0.4682
```

---

### 4. 训练历史记录

**修改位置**: 多处

**新增记录**:
- 初始化时添加 `cosine_sim_history = []` (第599行)
- 每个epoch记录 `cosine_sim_history.append(val_cosine_sim)` (第682行)
- 保存时包含 `'val_cosine_similarity': cosine_sim_history` (第739行)
- 恢复时加载 `cosine_sim_history` (第622行)

**保存的JSON结构**:
```json
{
  "train_loss": [...],
  "val_loss": [...],
  "val_cosine_similarity": [...],  // 新增
  "learnable_weights": {...},
  "config": {...}
}
```

---

### 5. 最终输出

**修改位置**: 第747-749行

**修改前**:
```python
print(f"Best validation loss: {best_val_loss:.6f}")
```

**修改后**:
```python
print(f"Best validation cosine loss: {best_val_loss:.6f}")
if cosine_sim_history:
    print(f"Best validation cosine similarity: {max(cosine_sim_history):.4f}")
```

---

## 📊 理解新指标

### Cosine Loss vs Cosine Similarity

| 指标 | 范围 | 优化目标 | 含义 |
|------|------|----------|------|
| **Cosine Loss** | [0, 2] | 最小化 | 1 - cosine_similarity |
| **Cosine Similarity** | [-1, 1] | 最大化 | 直接相似度 |

**关系**:
```python
cosine_loss = 1 - cosine_similarity

# 示例
cosine_similarity = 0.8
cosine_loss = 1 - 0.8 = 0.2  ✓ (loss越小越好)
```

### 评估标准

**Cosine Loss** (越小越好):
- `< 0.15`: 优秀 (cosine_sim > 0.85)
- `0.15-0.30`: 良好 (cosine_sim > 0.70)
- `0.30-0.50`: 中等 (cosine_sim > 0.50)
- `≈ 1.0`: 随机基线 (cosine_sim ≈ 0)
- `> 1.5`: 很差

**Cosine Similarity** (越大越好):
- `> 0.85`: 优秀
- `0.70-0.85`: 良好
- `0.50-0.70`: 中等
- `≈ 0.0`: 随机基线
- `< 0`: 异常 (学习错误方向)

---

## 🚀 使用方法

### 训练命令（无需修改）

```bash
python scripts/04_train_model.py \
    --embeddings_path data/processed/ligand_embeddings_256d.h5 \
    --output_dim 256 \
    --batch_size 32 \
    --num_epochs 300 \
    --lr 1e-3
```

### 监控训练

**关注的指标**:
1. **Val Cosine Sim** (主要指标) - 应该持续上升
2. **Val Cosine Loss** (优化目标) - 应该持续下降
3. **Val MSE** (参考) - 通常也会下降

**期望的训练轨迹**:
```
Epoch 1:   Cosine Sim: 0.05, Cosine Loss: 0.95, MSE: 1.90
Epoch 50:  Cosine Sim: 0.55, Cosine Loss: 0.45, MSE: 0.90
Epoch 150: Cosine Sim: 0.75, Cosine Loss: 0.25, MSE: 0.50
Epoch 300: Cosine Sim: 0.85, Cosine Loss: 0.15, MSE: 0.30
```

---

## 🔍 验证修改

### 快速测试

运行1个epoch验证代码是否正常：

```bash
python scripts/04_train_model.py \
    --num_epochs 1 \
    --batch_size 4 \
    --output_dir test_cosine_loss
```

**预期输出**:
```
Epoch 1/1
--------------------------------------------------
Train Loss: 0.xxxx
Val Cosine Loss: 0.xxxx, Val Cosine Sim: 0.xxxx, Val MSE: x.xxxx
Learning Rate: 1.00e-03
```

---

## 📈 与MSE的对比

### 理论期望

| 训练阶段 | MSE Loss | Cosine Loss | Cosine Sim |
|----------|----------|-------------|------------|
| 随机初始化 | ≈ 2.0 | ≈ 1.0 | ≈ 0.0 |
| 训练中期 | ≈ 0.8 | ≈ 0.4 | ≈ 0.6 |
| 训练后期 | ≈ 0.3 | ≈ 0.15 | ≈ 0.85 |

### 预期改进

使用Cosine Loss应该看到：
- ✅ **Cosine Similarity直接提升**: +5-10%
- ✅ **训练-评估一致性**: 优化目标 = 评估指标
- ✅ **检索性能提升**: 下游任务表现更好
- ✅ **收敛更稳定**: 方向对齐比距离对齐更鲁棒

---

## ⚠️ 注意事项

### 1. Early Stopping仍然有效

虽然从MSE改为Cosine Loss，但early stopping逻辑仍然正常工作：
- `val_loss` 现在是 `cosine_loss`
- 更小的cosine_loss = 更好的模型 ✓
- scheduler和best model保存都正常

### 2. MSE仍然被记录

保留了MSE用于对比：
- 可以验证理论关系: `MSE ≈ 2(1 - cosine_sim)`
- 便于与之前的MSE模型对比

### 3. 归一化很重要

Cosine Loss假设嵌入已归一化，确保：
- Ligand embeddings已经z-score归一化 ✓
- 模型输出可以是任意模长（cosine会自动归一化）

---

## 📚 相关文档

- **理论分析**: `docs/advanced_loss_functions.md`
- **快速开始**: `ADVANCED_LOSS_QUICK_START.md`
- **损失范围**: `docs/loss_metric_ranges.md`

---

## 🎯 下一步建议

### 1. 基线对比（推荐）

训练两个模型对比：
```bash
# MSE baseline (如果之前训练过，直接用结果)
# 或者切换回MSE重新训练

# Cosine Loss (新)
python scripts/04_train_model.py \
    --output_dir models/checkpoints_cosine_loss
```

对比指标：
- Validation Cosine Similarity (主要)
- Downstream retrieval accuracy (如果有)

### 2. 调优学习率（可选）

Cosine Loss可能需要略微调整学习率：
```bash
# 尝试略低的学习率
python scripts/04_train_model.py \
    --lr 5e-4  # 从1e-3降低到5e-4
```

### 3. 上InfoNCE（进阶）

如果Cosine Loss有提升，考虑升级到InfoNCE：
- 需要batch_size ≥ 32
- 预期再提升10-20%
- 参见 `models/advanced_losses.py`

---

## ✅ 总结

**修改完成**: ✅ `scripts/04_train_model.py` 已更新为使用Cosine Loss

**核心变化**:
- 训练损失: MSE → Cosine Loss
- 主要指标: MSE → Cosine Similarity
- 保留对比: 仍然记录MSE

**无需修改**:
- 命令行参数
- 模型结构
- 数据加载
- Early stopping逻辑

**立即开始训练**:
```bash
python scripts/04_train_model.py --batch_size 32 --num_epochs 300
```

预期看到cosine similarity持续上升，最终达到0.8+！🚀
