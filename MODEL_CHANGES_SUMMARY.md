# V3 模型简化修改总结

## 📋 修改清单

### ✅ 已完成修改

1. **移除角度消息传递中的 angle_deviation**
   - 文件: `models/improved_components.py`
   - 类: `GeometricAngleMessagePassing`
   - 修改: 只保留 `cos_angle`，移除 `angle_deviation` 计算

2. **标记物理约束loss为不推荐**
   - 文件: `scripts/04_train_model.py`
   - 修改: 在 `--use_physics_loss` 参数帮助文本中添加 `[NOT RECOMMENDED]` 标记
   - 默认值: 保持 `False`（默认禁用）

3. **梯度监控bug修复**
   - 文件: `scripts/04_train_model.py`
   - 修复: 自适应监控频率（前100个batch每10次，之后每50次）

---

## 📁 修改的文件

```
models/
└── improved_components.py         # 简化角度消息传递

scripts/
└── 04_train_model.py             # 标记物理loss不推荐 + 梯度监控修复

docs/
├── v3_model_simplifications.md   # 详细修改说明
├── gradient_monitoring_fix.md    # 梯度监控和残差连接分析
└── MODEL_CHANGES_SUMMARY.md      # 本文件
```

---

## 🔍 详细修改

### 修改 1: GeometricAngleMessagePassing

**位置**: `models/improved_components.py`

#### 输入维度变化

```python
# 修改前
input_dim = self.scalar_dim * 2 + angle_attr_dim
if use_geometry:
    input_dim += 2  # cos_angle + angle_deviation

# 修改后
input_dim = self.scalar_dim * 2 + angle_attr_dim
if use_geometry:
    input_dim += 1  # cos_angle only
```

#### Forward 函数变化

```python
# 修改前 (Lines 119-137)
cos_angle = ...
theta_eq_radians = triple_attr[:, 0] * math.pi
cos_eq = torch.cos(theta_eq_radians)
angle_deviation = cos_angle - cos_eq
angle_deviation_norm = angle_deviation / 2.0
angle_input.append(cos_angle.unsqueeze(-1))
angle_input.append(angle_deviation_norm.unsqueeze(-1))

# 修改后 (Lines 119-128)
cos_angle = ...
angle_input.append(cos_angle.unsqueeze(-1))
# angle_deviation 相关代码已移除
```

---

### 修改 2: 训练脚本物理loss标注

**位置**: `scripts/04_train_model.py:1289-1291`

```python
# 修改前
parser.add_argument("--use_physics_loss", action="store_true", default=False,
                    help="Enable physics constraint loss (bond/angle/dihedral energies)")

# 修改后
parser.add_argument("--use_physics_loss", action="store_true", default=False,
                    help="[NOT RECOMMENDED] Enable physics constraint loss (bond/angle/dihedral energies). "
                         "This adds extra complexity without clear benefits for representation learning.")
```

---

### 修改 3: 梯度监控频率

**位置**: `scripts/04_train_model.py:368-377`

```python
# 修改前
if monitor_gradients and batch_idx % 50 == 0:
    # 计算梯度范数
    print(f"  Batch {batch_idx}: Grad norm = ...")

# 修改后
if monitor_gradients:
    monitor_interval = 10 if batch_idx < 100 else 50
    if batch_idx % monitor_interval == 0:
        # 计算梯度范数
        print(f"  Batch {batch_idx}: Grad norm = ...")
```

---

## ⚠️ 重要提示

### 1. 向后兼容性破坏

**角度MP的输入维度发生变化**，旧的checkpoint无法直接加载！

```python
# 旧模型 checkpoint 加载会报错:
# RuntimeError: size mismatch for angle_mlp.0.weight:
# copying a param with shape torch.Size([64, OLD_DIM]) from checkpoint,
# the shape in current model is torch.Size([64, NEW_DIM])
```

**解决方案**: 从头训练新模型

### 2. 需要重新训练

如果你有已训练的V3模型：
- ❌ 无法直接使用旧checkpoint
- ✅ 需要使用新代码重新训练

### 3. 默认配置已改变

现在推荐的训练命令：

```bash
python scripts/04_train_model.py \
    --model_version v3 \
    --data_dir data/processed_pockets \
    --ligand_embeddings data/ligand_embeddings.h5 \
    --split_dir data/splits \
    --loss_fn cosine \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --use_amp \
    --monitor_gradients
    # 不要添加 --use_physics_loss
```

---

## 📊 修改前后对比

| 特性 | 修改前 | 修改后 | 影响 |
|------|--------|--------|------|
| **角度MP几何特征** | 2个 (cos + deviation) | 1个 (cos only) | 更简洁 |
| **角度MP输入维度** | scalar×2 + attr + 2 | scalar×2 + attr + 1 | -1维 |
| **物理loss** | 可选，无警告 | 可选，明确不推荐 | 更明确 |
| **梯度监控频率** | 每50个batch | 自适应 (10/50) | 更及时 |
| **模型复杂度** | 较高 | 中等 | ↓ |
| **训练稳定性** | 中 | 高 | ↑ |

---

## 🧪 测试修改

### 快速验证

```bash
# 1. 检查模型可以正常创建
python -c "from models.e3_gnn_encoder_v3 import RNAPocketEncoderV3; \
    model = RNAPocketEncoderV3(); \
    print('✓ Model created successfully')"

# 2. 运行debug脚本测试
python scripts/debug_training.py \
    --model_version v3 \
    --data_dir data/processed_pockets \
    --ligand_embeddings data/ligand_embeddings.h5 \
    --split_file data/splits/split_0.json \
    --num_batches 10

# 3. 短期训练测试
python scripts/04_train_model.py \
    --model_version v3 \
    --data_dir data/processed_pockets \
    --ligand_embeddings data/ligand_embeddings.h5 \
    --split_dir data/splits \
    --loss_fn cosine \
    --epochs 2 \
    --batch_size 16 \
    --monitor_gradients
```

**预期结果**:
- ✅ 模型正常创建和forward
- ✅ 梯度监控显示 batch 0, 10, 20, ... (不再只显示batch 0)
- ✅ 训练稳定，loss平滑下降
- ✅ 无 NaN/Inf 错误

---

## 📚 相关文档

- **详细修改说明**: `docs/v3_model_simplifications.md`
- **梯度问题诊断**: `docs/gradient_instability_diagnosis.md`
- **梯度监控修复**: `docs/gradient_monitoring_fix.md`
- **测试指南**: `docs/test_fixes_quickstart.md`

---

## 🤝 如果遇到问题

### 问题 1: 无法加载旧checkpoint

```python
# 错误信息
RuntimeError: Error(s) in loading state_dict for RNAPocketEncoderV3:
    size mismatch for angle_mp_layers.0.angle_mlp.0.weight: ...
```

**解决**: 从头训练，不要尝试加载旧checkpoint

---

### 问题 2: 训练不稳定

**检查清单**:
1. 确认已应用所有梯度稳定性修复（见 `docs/applied_fixes_summary.md`）
2. 确认没有使用 `--use_physics_loss`
3. 尝试降低学习率: `--lr 5e-5`
4. 尝试禁用AMP: 移除 `--use_amp`

---

### 问题 3: 梯度监控仍只显示batch 0

**检查**:
```bash
# 确认修复已应用
grep -A 3 "if monitor_gradients:" scripts/04_train_model.py | grep "monitor_interval"
# 应该看到: monitor_interval = 10 if batch_idx < 100 else 50
```

如果没有，重新应用修复（见 `docs/gradient_monitoring_fix.md`）

---

## ✅ 修改完成确认

- [x] `models/improved_components.py` - 角度MP简化
- [x] `scripts/04_train_model.py` - 物理loss标注 + 梯度监控
- [x] `docs/v3_model_simplifications.md` - 详细说明
- [x] `docs/gradient_monitoring_fix.md` - 监控修复说明
- [x] `MODEL_CHANGES_SUMMARY.md` - 本总结

**所有修改已完成！可以开始测试和训练。**

---

*修改完成时间: 2025-11-09*
*修改类型: 模型简化 + Bug修复*
*向后兼容: ❌ 不兼容，需重新训练*
