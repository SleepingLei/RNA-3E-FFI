# 先进损失函数 - 快速开始

**TL;DR**: 当前的MSE损失不适合嵌入对齐任务。使用InfoNCE + Cosine组合损失可以显著提升性能（预期+20-30%）。

---

## 🚀 快速使用

### 最简单：替换为Cosine Loss

```python
# 在 scripts/04_train_model.py 中
from models.advanced_losses import CosineSimilarityLoss

# 替换
# loss = F.mse_loss(pocket_embedding, target_embedding)

# 为
criterion = CosineSimilarityLoss()
loss = criterion(pocket_embedding, target_embedding)
```

**预期提升**: +5-10% cosine similarity

---

### 推荐：使用InfoNCE Loss

```python
from models.advanced_losses import InfoNCELoss

# 初始化（在训练脚本顶部）
criterion = InfoNCELoss(temperature=0.07)

# 在训练循环中
loss = criterion(pocket_embedding, target_embedding)
```

**重要**: InfoNCE需要较大batch size (≥32)

**预期提升**: +15-25% retrieval accuracy

---

### 最佳：Multi-Stage Training

```python
from models.advanced_losses import MultiStageScheduler

# 初始化scheduler
loss_scheduler = MultiStageScheduler(
    stage1_epochs=50,   # MSE warmup
    stage2_epochs=100,  # Contrastive learning
    stage3_start=150    # Fine-tuning
)

# 在每个epoch开始时
criterion, stage_name = loss_scheduler.get_loss(epoch)
recommended_lr = loss_scheduler.get_recommended_lr()

# 使用
loss = criterion(pocket_embedding, target_embedding)
```

**预期提升**: +20-30% 总体性能

---

## 📊 对比实验（建议）

### 实验1: Baseline (当前)
```bash
python scripts/04_train_model.py \
    --loss_fn mse \
    --lr 1e-3 \
    --batch_size 16 \
    --num_epochs 300
```

### 实验2: Cosine Loss
```bash
python scripts/04_train_model.py \
    --loss_fn cosine \
    --lr 1e-3 \
    --batch_size 16 \
    --num_epochs 300
```

### 实验3: InfoNCE Loss
```bash
python scripts/04_train_model.py \
    --loss_fn infonce \
    --lr 5e-4 \
    --batch_size 32 \
    --temperature 0.07 \
    --num_epochs 300
```

### 实验4: Combined Loss (推荐)
```bash
python scripts/04_train_model.py \
    --loss_fn combined \
    --alpha 0.5 \
    --beta 0.3 \
    --gamma 0.2 \
    --lr 5e-4 \
    --batch_size 32 \
    --num_epochs 300
```

### 实验5: Multi-Stage (最佳)
```bash
python scripts/04_train_model.py \
    --loss_fn multistage \
    --stage1_epochs 50 \
    --stage2_epochs 100 \
    --batch_size 32 \
    --num_epochs 300
```

---

## 🎯 关键参数

### InfoNCE Temperature (τ)

```python
temperature = 0.07  # CLIP默认，通常是好的起点
```

- **0.01-0.05**: Very hard - 梯度大，学习快，但可能不稳定
- **0.07-0.10**: Balanced - 推荐
- **0.10-0.20**: Soft - 梯度小，学习慢，但更稳定

### Batch Size

```
16   ❌ 太小（负样本不足）
32   ✅ 推荐最小值
64   ✅✅ 更好
128+ ✅✅✅ 最佳（如果内存允许）
```

如果内存不足，使用**梯度累积**：
```python
accumulation_steps = 2  # effective_batch_size = 16 * 2 = 32

for i, batch in enumerate(dataloader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 组合损失权重

```python
alpha = 0.5  # InfoNCE - 主导（判别性学习）
beta = 0.3   # Cosine - 辅助（方向对齐）
gamma = 0.2  # MSE - 保留（距离约束）
```

---

## 📈 评估指标

### 训练时监控

```python
# 1. Loss值
train_loss = ...
val_loss = ...

# 2. Cosine Similarity（最重要！）
avg_cosine_sim = F.cosine_similarity(pred, target).mean()

# 3. InfoNCE准确率（如果使用InfoNCE）
# 在batch内，正确配体是否rank第一
```

### 验证时评估

```python
from models.advanced_losses import compute_retrieval_metrics

metrics = compute_retrieval_metrics(
    pocket_embeddings,
    ligand_embeddings,
    top_k=[1, 5, 10]
)

print(f"Top-1: {metrics['top1_accuracy']}")
print(f"Top-5: {metrics['top5_accuracy']}")
print(f"MRR: {metrics['mrr']}")
```

---

## ⚠️ 常见问题

### Q1: InfoNCE loss很高（>10）

**原因**: 温度太小或batch size太小

**解决**:
- 增大temperature: 0.07 → 0.10
- 增大batch size: 16 → 32

### Q2: Cosine similarity没有提升

**原因**: 可能是学习率或batch size问题

**解决**:
- 降低学习率: 1e-3 → 5e-4
- 使用warmup
- 检查数据是否正确归一化

### Q3: 内存不足（batch size无法增大）

**解决方案**:
1. 使用梯度累积（见上）
2. 减小embedding维度（1536 → 256）
3. 使用mixed precision training
   ```python
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()
   with autocast():
       loss = model(batch)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

### Q4: 如何处理假负样本？

**问题**: 在InfoNCE中，(pocket_A, ligand_B) 被当作负样本，但ligand_B可能也能结合pocket_A

**解决**:
1. 使用 `SupervisedContrastiveLoss` 并提供配体相似度矩阵
2. 过滤高相似度配体（>0.9）
3. 使用soft labels

---

## 📁 文件说明

```
models/advanced_losses.py          ← 所有损失函数实现
docs/advanced_loss_functions.md   ← 完整技术文档
ADVANCED_LOSS_QUICK_START.md      ← 本文件（快速开始）
scripts/04_train_model.py          ← 需要更新以支持新损失
```

---

## 🎓 理论背景（简版）

### 为什么MSE不好？

```python
pred   = [1, 0, 0]  # 方向对
target = [2, 0, 0]  # 只是模长不同

MSE = 1.0 ❌ 高loss
Cosine Similarity = 1.0 ✅ 完美对齐
```

MSE关心绝对距离，但在嵌入空间中，**方向比距离更重要**。

### 为什么InfoNCE好？

InfoNCE同时优化：
1. **正样本接近**: pocket_i ↔ ligand_i
2. **负样本远离**: pocket_i ↮ ligand_j (j≠i)

这学习到的是**判别性表示**，不仅对齐，而且可区分。

**成功案例**: CLIP (图像-文本), ProteinCLIP (蛋白质-配体)

---

## 🚦 实施路线图

### 阶段1: 快速验证 (1-2天)

1. ✅ 实现Cosine Loss
2. ✅ 训练一个小实验（50 epochs）
3. ✅ 对比MSE baseline

**决策点**: 如果Cosine有提升 → 继续；否则检查数据/超参数

### 阶段2: InfoNCE实验 (3-5天)

1. ✅ 实现InfoNCE Loss
2. ✅ 调优batch size和temperature
3. ✅ 完整训练（300 epochs）
4. ✅ 评估retrieval metrics

**决策点**: 如果InfoNCE显著提升 → 进入阶段3

### 阶段3: 组合优化 (1周)

1. ✅ 实现Multi-Stage Training
2. ✅ 调优权重和阶段划分
3. ✅ 消融实验
4. ✅ 最终模型选择

---

## 📞 需要帮助？

查看完整文档：`docs/advanced_loss_functions.md`

测试损失函数：
```bash
python models/advanced_losses.py
```

---

**总结**:
- 🥉 简单但有效：Cosine Loss (+5-10%)
- 🥈 强力提升：InfoNCE Loss (+15-25%)
- 🥇 最佳方案：Multi-Stage Combined (+20-30%)

开始实验吧！🚀
