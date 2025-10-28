# CUDA内存优化详解

## 🔍 问题分析：为什么第5个Epoch才OOM？

### 原始错误信息分析
```
CUDA out of memory. Tried to allocate 4.15 GiB.
GPU 0 has a total capacity of 44.53 GiB
- Free: 3.84 GiB
- In use by process: 40.68 GiB
- Allocated by PyTorch: 36.78 GiB
- Reserved but unallocated: 3.56 GiB
```

### 关键问题

#### 1. **为什么前4个epoch正常，第5个才崩溃？**

**主要原因：内存累积效应**

- **Epoch 1-2**: GPU缓存在"学习"数据访问模式，内存使用逐渐增加
- **Epoch 3-4**: PyTorch缓存分配器开始产生碎片，保留内存增加
- **Epoch 5**: 碎片化严重 + 遇到大样本 → OOM

#### 2. **内存泄漏的具体来源**

```python
# ❌ 问题代码
def __getitem__(self, idx):
    data = torch.load(graph_path)  # 每次都从磁盘加载
    data.y = self.ligand_embeddings[key]  # 直接引用，可能保持引用
    return data
```

**问题：**
- `torch.load()` 每次调用都会创建新的Python对象和CUDA tensor
- 重复调用会让Python解释器保留一些内部状态
- 随着epoch增加，这些"幽灵"引用累积

#### 3. **PyTorch缓存分配器的碎片化**

PyTorch使用缓存分配器来提高性能：
- 不会立即释放GPU内存给OS
- 保留内存块以备复用
- 多个epoch后，内存变得碎片化

```
Epoch 1:  [====] [====] [====]  (整齐的内存块)
Epoch 3:  [==][=][===][=][==]  (开始碎片化)
Epoch 5:  [=][==][=][=][===]  (严重碎片化，无法分配大块)
```

#### 4. **DataLoader的pin_memory陷阱**

```python
# ❌ 原来的配置
train_loader = DataLoader(
    dataset,
    num_workers=4,
    pin_memory=True  # 每个worker都会预加载数据到固定内存
)
```

- `pin_memory=True` 会将CPU数据固定到内存，快速传输到GPU
- 但会额外占用 `batch_size × num_workers` 的内存
- 4个workers × batch_size=2 = 同时8个样本固定在内存中

## ✅ 已应用的优化方案

### 1. **修复数据加载器的内存泄漏**

```python
# ✅ 优化后
def __getitem__(self, idx):
    data = torch.load(graph_path, weights_only=False)

    # 创建新tensor而不是直接引用
    data.y = torch.tensor(ligand_embedding, dtype=torch.float32)

    return data
```

**改进：**
- 使用 `torch.tensor()` 创建独立副本
- 避免共享引用导致的内存保持

### 2. **优化训练循环的内存管理**

```python
# ✅ 优化后的训练循环
for batch_idx, batch in enumerate(loader):
    batch = batch.to(device)

    # Forward & Backward
    pocket_embedding = model(batch)
    loss = F.mse_loss(pocket_embedding, target)
    loss.backward()
    optimizer.step()

    # 记录loss值
    total_loss += loss.item()

    # ⭐ 关键：显式删除中间变量
    del pocket_embedding, target_embedding, loss

    # ⭐ 每10个batch清理一次缓存（平衡性能和内存）
    if (batch_idx + 1) % 10 == 0:
        torch.cuda.empty_cache()
```

**改进：**
- 显式 `del` 删除中间变量，立即释放引用
- 减少 `empty_cache()` 调用频率（原来每个batch，现在每10个）
- 避免过度调用 `empty_cache()` 的性能损失

### 3. **优化DataLoader配置**

```python
# ✅ 优化后
train_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=min(args.num_workers, 2),  # 限制最多2个worker
    pin_memory=False,  # 禁用pin_memory节省内存
    persistent_workers=False  # 不在epoch之间保持workers
)
```

**改进：**
- `num_workers=2`: 减少同时预加载的数据量
- `pin_memory=False`: 节省固定内存（对于小batch影响不大）
- `persistent_workers=False`: 每个epoch后关闭workers

### 4. **Epoch级别的内存管理**

```python
# ✅ 在epoch开始和结束时清理
for epoch in range(num_epochs):
    # Epoch开始：清理缓存 + 同步
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 训练...
    train_epoch(...)

    # 验证...
    evaluate(...)

    # Epoch结束：再次清理
    if device.type == 'cuda':
        torch.cuda.empty_cache()
```

**改进：**
- `torch.cuda.synchronize()`: 确保所有CUDA操作完成
- 在epoch边界清理，减少碎片累积

## 📊 CUDA内存使用的关键概念

### 1. **PyTorch内存分配器的三层结构**

```
┌─────────────────────────────────────┐
│  PyTorch Allocated Memory (36.78G)  │  ← 实际分配给tensor的
├─────────────────────────────────────┤
│  Reserved but Unallocated (3.56G)   │  ← PyTorch缓存池
├─────────────────────────────────────┤
│  Free GPU Memory (3.84G)            │  ← OS层面可用内存
└─────────────────────────────────────┘
```

- **Allocated**: tensor实际占用的内存
- **Reserved**: PyTorch缓存起来的，没还给OS
- **Free**: 真正可用的GPU内存

### 2. **torch.cuda.empty_cache() 的作用**

```python
torch.cuda.empty_cache()
```

- ❌ **不会**释放tensor占用的内存
- ✅ **会**将Reserved内存还给OS
- ⚠️ **代价**：下次分配需要重新向OS申请（慢）

**何时使用：**
- Epoch之间（内存需求可能变化）
- 批量推理后（释放大块内存）
- ❌ 不要在每个batch后使用（性能损失大）

### 3. **内存碎片化示意图**

```
良好状态（Epoch 1）:
┌────────┬────────┬────────┬────────┐
│ Batch1 │ Batch2 │ Batch3 │ Batch4 │
└────────┴────────┴────────┴────────┘

碎片化（Epoch 5）:
┌──┬─┬───┬─┬──┬─┬────┬──┬─┬───┬──┐
│B1│ │B2 │ │B3│ │B4  │  │ │...│  │
└──┴─┴───┴─┴──┴─┴────┴──┴─┴───┴──┘
      ↑空洞         ↑空洞     ↑空洞

问题：即使总空闲内存足够，也无法分配大块连续内存！
```

### 4. **梯度累积的内存影响**

```python
# 梯度会一直保留直到optimizer.zero_grad()
loss.backward()  # 创建梯度tensor
optimizer.step()  # 使用梯度
optimizer.zero_grad()  # ⭐ 释放梯度内存
```

## 🚀 进一步优化建议

### 1. **使用梯度累积减小batch size**

如果仍然OOM，可以使用梯度累积：

```python
accumulation_steps = 4  # 累积4个batch相当于batch_size × 4

for batch_idx, batch in enumerate(loader):
    loss = train_step(batch)
    loss = loss / accumulation_steps  # 平均梯度
    loss.backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**效果：**
- batch_size=2, accumulation_steps=4 → 等效batch_size=8
- 但每次只需加载2个样本的内存

### 2. **混合精度训练（FP16）**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in loader:
    with autocast():  # 自动使用FP16
        output = model(batch)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

**效果：**
- 内存使用减半（FP16 vs FP32）
- 速度提升1.5-2x
- 几乎无精度损失（对于大多数任务）

### 3. **使用gradient checkpointing**

```python
# 在模型中启用
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def forward(self, x):
        # 不保存中间激活值
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x
```

**效果：**
- 减少激活值内存占用（50-70%）
- 代价：反向传播时需要重新计算（速度降低20-30%）

### 4. **监控内存使用**

```python
def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# 在关键位置调用
print_gpu_memory()  # Epoch开始
train_epoch(...)
print_gpu_memory()  # Epoch结束
```

### 5. **使用环境变量优化内存分配**

```bash
# 减少内存碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 在脚本中设置
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

## 🎯 推荐的训练配置

### 对于44GB GPU（如A100）：

```bash
python scripts/04_train_model.py \
    --batch_size 2 \              # 保持较小batch
    --num_workers 2 \              # 限制workers
    --use_multi_hop \
    --use_nonbonded \
    --use_layer_norm \
    --num_epochs 300
```

### 如果仍然OOM：

```bash
# 方案1：进一步减小batch size
--batch_size 1

# 方案2：使用梯度累积
--batch_size 1 --gradient_accumulation_steps 4

# 方案3：使用混合精度（需要代码支持）
--use_amp
```

## 📝 总结

### OOM的根本原因：
1. ✅ **内存泄漏**: `torch.load()` 重复调用 + 引用共享
2. ✅ **缓存碎片化**: 多个epoch后PyTorch缓存池碎片化
3. ✅ **DataLoader配置**: 过多workers + pin_memory占用额外内存
4. ✅ **缺少清理**: 中间变量没有显式释放

### 已修复的关键点：
- ✅ 数据加载时创建独立tensor副本
- ✅ 显式删除中间变量
- ✅ 减少workers和禁用pin_memory
- ✅ 周期性清理CUDA缓存
- ✅ Epoch边界同步和清理

### 内存优化优先级：
1. **必须做**: 修复内存泄漏（已完成）
2. **应该做**: 优化DataLoader配置（已完成）
3. **可选做**: 混合精度训练（性能提升）
4. **最后手段**: Gradient checkpointing（牺牲速度）

现在重新运行训练应该不会再OOM了！
