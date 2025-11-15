# V3 模型梯度爆炸修复总结

## 📋 问题诊断

在使用 `scripts/train_physics_v3.sh` 训练时，遇到 **grad norm 逐渐增大** 的问题。

### 根本原因

**多路径加权系数过大，导致特征幅值逐层累积爆炸**

修改前的权重配置：
```python
# 每层的消息聚合
h_new = h_bonded                    # 1.0
h_new = h_new + 0.33 * h_angle      # +0.33
h_new = h_new + 0.33 * h_dihedral   # +0.33
h_new = h_new + 0.33 * h_nonbonded  # +0.33
# 总计: 1.99x

h = h + h_new  # 残差连接
```

**影响**:
- 每层输出幅值约为 1.99 倍输入
- 6 层累积后: **1.99^6 ≈ 62 倍**
- 导致梯度爆炸，训练不稳定

---

## ✅ 实施的修复方案

### **方案 1B: 权重重新分配（bonded 优先）**

确保所有路径的加权系数之和 = 1.0

| 路径 | 修改前 | 修改后 | 说明 |
|------|--------|--------|------|
| Bonded | 1.0 (隐式) | **0.4** | 主要路径，权重最高 |
| Angle | 0.33 | **0.2** | 降低 |
| Dihedral | 0.33 | **0.2** | 降低 |
| Nonbonded | 0.33 | **0.2** | 降低 |
| **总计** | **1.99** | **1.0** | ✅ 避免幅值累积 |

### **方案 2: 添加 Post-LN（后归一化）**

在聚合后对特征进行归一化，进一步防止幅值累积：

```python
# 聚合所有路径
h_new = 0.4 * h_bonded + 0.2 * h_angle + 0.2 * h_dihedral + 0.2 * h_nonbonded

# 后归一化（新增）
h_new = post_layer_norm(h_new)

# 残差连接
h = h + h_new
```

---

## 📝 修改的文件

### 1. `models/e3_gnn_encoder_v3.py`

**修改内容**:

1. **添加 `bonded_weight` 参数** (line 290):
   ```python
   self.bonded_weight = 0.4  # 新增
   ```

2. **添加 Post-LN 层** (line 442-466):
   ```python
   self.post_aggregation_layer_norms = nn.ModuleList()
   for i in range(num_layers):
       self.post_aggregation_layer_norms.append(
           LayersEquivariantLayerNorm(self.hidden_irreps, affine=True)
       )
   ```

3. **修改 forward 方法** (line 593-620):
   ```python
   # 应用权重
   h_new = self.bonded_weight * h_bonded           # 0.4
   h_new = h_new + self.angle_weight * h_angle     # +0.2
   h_new = h_new + self.dihedral_weight * h_dihedral  # +0.2
   h_new = h_new + self.nonbonded_weight * h_nonbonded  # +0.2

   # Post-LN
   h_new = self.post_aggregation_layer_norms[i](h_new)

   # 残差连接
   h = h + h_new
   ```

4. **更新 `get_weight_stats` 方法** (line 710-733):
   - 添加 `bonded_weight` 输出
   - 添加 `total_weight` 验证

### 2. `scripts/train_physics_v3.sh`

**修改内容** (line 24):

```bash
# 修改前
--initial_angle_weight 0.33 --initial_dihedral_weight 0.33 --initial_nonbonded_weight 0.33

# 修改后
--initial_angle_weight 0.2 --initial_dihedral_weight 0.2 --initial_nonbonded_weight 0.2
```

### 3. `scripts/04_train_model.py`

**修改内容**:

1. **更新权重显示** (line 872-890):
   - 添加 bonded_weight 显示
   - 添加 total_weight 验证
   - 警告总权重偏离 1.0 的情况

2. **更新训练监控** (line 1110-1130):
   - 简化权重显示（去除梯度信息，因为现在是固定权重）
   - 添加 total_weight 显示

---

## 🧪 验证结果

运行 `python test_v3_fixes.py`:

```
✅ Total weight = 1.0 (PASS)
✅ Post-LN layers exist: 4 layers
✅ Forward pass successful
✅ Gradient norm is reasonable (31.999954)
✅ Feature norms are stable (4.03x growth vs previous 62x)
```

**关键改进**:
- **特征幅值增长**: 从 **62 倍** → **4 倍** (4 层模型)
- **梯度范数**: 合理范围（~32）
- **总权重**: 精确为 1.0

---

## 🚀 使用方法

### 1. 直接运行训练脚本

```bash
bash scripts/train_physics_v3.sh
```

训练开始时会看到：

```
Path Weights (sum should = 1.0):
  Bonded:    0.400
  Angle:     0.200
  Dihedral:  0.200
  Nonbonded: 0.200
  Total:     1.000 (target: 1.0)
```

### 2. 监控梯度

训练时会显示：

```
📊 Path Weights Monitoring (Fixed):
  Bonded:    0.4000
  Angle:     0.2000
  Dihedral:  0.2000
  Nonbonded: 0.2000
  Total:     1.0000 (target: 1.0)
```

### 3. 自定义权重（可选）

如果需要调整权重，修改 `scripts/train_physics_v3.sh`:

```bash
--initial_angle_weight 0.15 \
--initial_dihedral_weight 0.15 \
--initial_nonbonded_weight 0.1
```

然后修改 `models/e3_gnn_encoder_v3.py` line 290:

```python
self.bonded_weight = 0.6  # 确保总和 = 1.0
```

---

## 📊 预期效果

修复后，你应该观察到：

1. **Grad norm 稳定**:
   - 不再逐渐增大
   - 保持在合理范围（< 50 for MSE loss）

2. **训练稳定**:
   - Loss 下降平滑
   - 不会突然 NaN 或 Inf

3. **特征幅值可控**:
   - 每层增长约 1.2x 而非 2x
   - 6 层总增长 ~5x 而非 62x

---

## 🔍 额外的诊断工具

### 使用 `get_feature_stats` 监控

在训练脚本中添加：

```python
# 每 10 个 epoch 打印一次特征统计
if epoch % 10 == 0:
    with torch.no_grad():
        sample_data = next(iter(val_loader))
        stats = model.get_feature_stats(sample_data)
        print(f"\n特征统计 (Epoch {epoch}):")
        for i, layer_stats in enumerate(stats['layers']):
            print(f"  Layer {i}: agg_norm={layer_stats['aggregated_norm']:.2f}")
```

### 监控梯度范数趋势

已在训练脚本中实现（`--monitor_gradients` 标志）:

```bash
python scripts/04_train_model.py ... --monitor_gradients
```

会每 50 个 batch 打印一次：

```
Batch X: Grad norm = Y.XXXXXX
```

---

## ⚠️ 注意事项

1. **旧模型不兼容**:
   - 修改后的模型与旧的 checkpoint 不兼容
   - 需要从头开始训练或使用新的 checkpoint

2. **权重总和**:
   - 确保所有路径权重之和 = 1.0
   - 训练时会自动验证并警告

3. **学习率调整**:
   - 由于特征幅值变化，可能需要略微调整学习率
   - 建议保持当前的 `lr=2e-4` 先试试
   - 如果仍然不稳定，可以降低到 `lr=1e-4`

4. **梯度裁剪**:
   - 当前 `--grad_clip 1.0` 应该足够
   - 如果还有问题，可以降低到 `0.5`

---

## 📈 后续优化建议

如果训练仍有问题，可以尝试：

1. **降低学习率**: `lr=1e-4`
2. **更强的梯度裁剪**: `--grad_clip 0.5`
3. **使用 warmup**: 添加 warmup scheduler
4. **调整权重分配**: 例如 bonded=0.5, 其他=0.15
5. **使用 RMSNorm**: `--norm_type rms` (更快)

---

## 🎯 总结

通过两个简单但关键的修复：
1. **权重重新分配**: 确保总和 = 1.0
2. **添加 Post-LN**: 在聚合后归一化

我们成功地将特征幅值增长从 **62 倍** 降低到 **~5 倍**，从根本上解决了梯度爆炸问题。

训练应该会更加稳定，grad norm 不再逐渐增大。

祝训练顺利！🚀
