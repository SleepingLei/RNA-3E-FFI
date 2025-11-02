# 损失函数使用指南

**更新日期**: 2025-11-02
**版本**: 支持多种损失函数的灵活配置

---

## 🎯 概述

训练脚本现在支持通过命令行参数灵活指定损失函数，无需修改代码。

支持的损失函数：
1. **MSE** - Mean Squared Error（基线）
2. **Cosine** - Cosine Similarity Loss（推荐）
3. **Cosine+MSE** - 组合损失
4. **InfoNCE** - 对比学习损失（高级）

---

## 📝 命令行参数

### 核心参数

```bash
--loss_fn {mse,cosine,cosine_mse,infonce}
```
选择损失函数类型（默认: `cosine`）

### 组合损失参数（仅用于 `cosine_mse`）

```bash
--cosine_weight FLOAT    # Cosine loss权重（默认: 0.7）
--mse_weight FLOAT       # MSE loss权重（默认: 0.3）
```

### InfoNCE参数（仅用于 `infonce`）

```bash
--temperature FLOAT      # 温度参数（默认: 0.07）
```

---

## 🚀 使用示例

### 1️⃣ MSE Loss（基线）

```bash
python scripts/04_train_model.py \
    --loss_fn mse \
    --batch_size 16 \
    --num_epochs 300 \
    --output_dir models/checkpoints_mse
```

**特点**：
- ✅ 简单直接
- ❌ 与下游任务（cosine similarity）不一致
- ❌ 对异常值敏感

**适用场景**: 作为baseline对比

---

### 2️⃣ Cosine Loss（推荐）

```bash
python scripts/04_train_model.py \
    --loss_fn cosine \
    --batch_size 16 \
    --num_epochs 300 \
    --output_dir models/checkpoints_cosine
```

**特点**：
- ✅ 与下游任务一致（都用cosine similarity）
- ✅ 方向对齐，尺度不变
- ✅ 适合归一化嵌入
- ✅ 训练稳定

**适用场景**: **默认选择**，适合大多数情况

**预期效果**: 比MSE提升5-10% cosine similarity

---

### 3️⃣ Cosine + MSE 组合（实验性）

```bash
python scripts/04_train_model.py \
    --loss_fn cosine_mse \
    --cosine_weight 0.7 \
    --mse_weight 0.3 \
    --batch_size 16 \
    --num_epochs 300 \
    --output_dir models/checkpoints_combined
```

**特点**：
- ✅ 同时优化方向（cosine）和距离（MSE）
- ⚠️ 需要调优权重参数
- ⚠️ 收益可能有限（两者高度相关）

**权重建议**：
- 主要优化方向: `--cosine_weight 0.8 --mse_weight 0.2`
- 平衡优化: `--cosine_weight 0.7 --mse_weight 0.3`（默认）
- 主要优化距离: `--cosine_weight 0.5 --mse_weight 0.5`

**适用场景**: 想要同时约束方向和距离

---

### 4️⃣ InfoNCE Loss（高级，最推荐）

```bash
python scripts/04_train_model.py \
    --loss_fn infonce \
    --temperature 0.07 \
    --batch_size 32 \
    --num_epochs 300 \
    --lr 5e-4 \
    --output_dir models/checkpoints_infonce
```

**关键要求**：
- ⚠️ **batch_size ≥ 16**（推荐 ≥ 32）
- ⚠️ 学习率可能需要略微调低（1e-3 → 5e-4）

**特点**：
- ✅ CLIP、SimCLR等的核心损失
- ✅ 同时优化正样本对齐 + 负样本分离
- ✅ 学习判别性表示
- ✅ 最适合检索任务

**温度参数调优**：
- `0.05`: Hard, 梯度大，学习快但可能不稳定
- `0.07`: **Balanced**（CLIP默认，推荐）
- `0.10`: Soft, 梯度小，学习慢但更稳定

**适用场景**: 下游任务是检索/排序，且batch size足够大

**预期效果**: 比Cosine再提升10-20% retrieval accuracy

---

## 📊 输出指标说明

训练过程中会自动输出相应的指标：

### MSE Loss
```
Train Loss: 0.8234, Cosine Sim: 0.5912
Val Loss: 0.7891, Cosine Sim: 0.6109, MSE: 0.7891
```
- **Loss**: MSE值
- **Cosine Sim**: 监控指标（不用于优化）

### Cosine Loss
```
Train Loss: 0.3456, Cosine Sim: 0.6544
Val Loss: 0.3201, Cosine Sim: 0.6799, MSE: 0.6402
```
- **Loss**: Cosine loss (1 - cosine_similarity)
- **Cosine Sim**: 主要指标
- **MSE**: 参考对比

### Cosine+MSE Loss
```
Train Loss: 0.4123, Cosine Sim: 0.6234
Val Loss: 0.3987, Cosine Sim: 0.6456, MSE: 0.7123
```
- **Loss**: 组合损失 (α*cosine + β*MSE)
- 会在日志中记录各组件的权重

### InfoNCE Loss
```
Train Loss: 3.2145, Cosine Sim: 0.7234, InfoNCE Acc: 28.12%
Val Loss: 3.1234, Cosine Sim: 0.7456, MSE: 0.4512, InfoNCE Acc: 31.25%
```
- **Loss**: InfoNCE值（通常>1）
- **InfoNCE Acc**: batch内检索准确率（越高越好）
- **Cosine Sim**: 监控指标
- **MSE**: 参考对比

---

## 🎯 推荐流程

### 阶段1: 快速验证（1-2天）

对比MSE和Cosine：

```bash
# Baseline
python scripts/04_train_model.py \
    --loss_fn mse \
    --num_epochs 100 \
    --output_dir models/baseline_mse

# Cosine
python scripts/04_train_model.py \
    --loss_fn cosine \
    --num_epochs 100 \
    --output_dir models/test_cosine
```

**决策**: 如果Cosine有提升 → 进入阶段2

---

### 阶段2: InfoNCE实验（3-5天）

```bash
python scripts/04_train_model.py \
    --loss_fn infonce \
    --temperature 0.07 \
    --batch_size 32 \
    --lr 5e-4 \
    --num_epochs 300 \
    --output_dir models/test_infonce
```

**决策**: 如果InfoNCE显著提升 → 进入阶段3

---

### 阶段3: 调优（1周）

调优InfoNCE超参数：
- Temperature: 0.05, 0.07, 0.10
- Batch size: 32, 64
- Learning rate: 5e-4, 7e-4, 1e-3

---

## ⚠️ 常见问题

### Q1: InfoNCE loss很高（>5）

**A**: 这是正常的！InfoNCE loss的尺度与MSE/Cosine不同。
- 初始值通常在3-5之间
- 关注趋势（下降）而非绝对值
- 关注InfoNCE Accuracy（应该>batch_size倒数）

---

### Q2: InfoNCE Accuracy很低（<10%）

**A**: 可能的原因：
1. Batch size太小 → 增大到32+
2. Temperature太小 → 增大到0.10
3. 模型刚开始训练 → 继续训练

**随机baseline**: 1/batch_size
- batch_size=32 → random accuracy=3.125%
- batch_size=64 → random accuracy=1.56%

---

### Q3: 不同loss function的模型能对比吗？

**A**: 可以！应该对比的指标：
- ✅ **Cosine Similarity**（主要）- 都会记录
- ✅ **Downstream retrieval accuracy** - 如果有
- ❌ Loss值本身 - 不可比（尺度不同）

---

### Q4: 如何选择batch size？

| Loss Function | 最小Batch | 推荐Batch | 原因 |
|--------------|----------|----------|------|
| MSE | 4+ | 16+ | 无特殊要求 |
| Cosine | 4+ | 16+ | 无特殊要求 |
| Cosine+MSE | 4+ | 16+ | 无特殊要求 |
| **InfoNCE** | **16+** | **32+** | 需要足够负样本 |

---

### Q5: 如何恢复训练？

所有loss function都支持恢复训练：

```bash
python scripts/04_train_model.py \
    --resume \
    --checkpoint models/checkpoints_xxx/best_model.pt \
    --loss_fn cosine  # 必须与原训练一致！
```

⚠️ **重要**: `--loss_fn` 必须与原训练时一致！

---

## 📈 实验对比模板

创建实验对比脚本：

```bash
#!/bin/bash

# MSE Baseline
python scripts/04_train_model.py \
    --loss_fn mse \
    --batch_size 16 \
    --num_epochs 300 \
    --output_dir models/exp_mse

# Cosine
python scripts/04_train_model.py \
    --loss_fn cosine \
    --batch_size 16 \
    --num_epochs 300 \
    --output_dir models/exp_cosine

# Cosine+MSE
python scripts/04_train_model.py \
    --loss_fn cosine_mse \
    --cosine_weight 0.7 \
    --mse_weight 0.3 \
    --batch_size 16 \
    --num_epochs 300 \
    --output_dir models/exp_combined

# InfoNCE
python scripts/04_train_model.py \
    --loss_fn infonce \
    --temperature 0.07 \
    --batch_size 32 \
    --lr 5e-4 \
    --num_epochs 300 \
    --output_dir models/exp_infonce
```

---

## 🎓 理论背景

详细分析参见：
- `docs/advanced_loss_functions.md` - 完整技术文档
- `docs/loss_metric_ranges.md` - 指标范围分析
- `ADVANCED_LOSS_QUICK_START.md` - 快速开始

---

## ✅ 总结

**简单场景**: 使用 `--loss_fn cosine`（默认）

**追求性能**: 使用 `--loss_fn infonce` + `--batch_size 32`

**实验对比**: 依次尝试 mse → cosine → infonce

**关键原则**: 训练时优化的指标应该与下游任务使用的指标一致！

---

**立即开始**:
```bash
# 推荐命令（Cosine Loss）
python scripts/04_train_model.py \
    --loss_fn cosine \
    --batch_size 32 \
    --num_epochs 300
```
