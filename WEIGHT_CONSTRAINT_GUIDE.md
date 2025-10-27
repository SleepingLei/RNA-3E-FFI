# 可学习权重约束指南

## 📋 概述

本指南介绍如何使用 **RNAPocketEncoderV2Fixed**，这是一个带权重约束的模型版本，可以防止可学习权重在训练过程中归零。

---

## 🔍 问题背景

### 为什么权重会归零？

在 v2.0 模型中，我们有三个可学习权重：
- `angle_weight`: 角度路径贡献权重
- `dihedral_weight`: 二面角路径贡献权重
- `nonbonded_weight`: 非键相互作用权重

**归零场景**：
```python
# 1. 数据缺失多跳路径
if not hasattr(data, 'triple_index'):
    h_angle = 0  # 角度贡献为 0

# 2. 前向传播
h_new = h_bonded + angle_weight * 0  # angle_weight 对损失无影响

# 3. 反向传播
angle_weight.grad ≈ -0.0001 或 0  # 梯度很小或为负

# 4. 优化器更新
# Adam 优化器逐步减小权重
angle_weight: 0.5 → 0.49 → 0.48 → ... → 0.0
```

---

## ✅ 解决方案：权重约束

### 标准版本 vs 约束版本

| 特性 | 标准版本 | 约束版本 (Fixed) |
|-----|---------|-----------------|
| 参数化 | `weight = nn.Parameter(0.5)` | `log_weight = nn.Parameter(log(0.5))` |
| 权重值 | 可以 → 0 | 永远 > 0 |
| 梯度 | 直接更新权重 | 更新 log(权重) |
| 稳定性 | ❌ 可能崩溃 | ✅ 数学保证 |

### 数学原理

**约束版本使用对数空间参数化**：

```
标准版本:
  weight = w
  更新: w ← w - lr * grad
  问题: 如果 grad > 0 持续，w 可能 → 0

约束版本:
  weight = exp(log_w)
  更新: log_w ← log_w - lr * grad
  优点: exp(log_w) > 0 对所有 log_w 成立
```

**示例**：
```python
# 标准版本
w = 0.5
grad = 0.01
for _ in range(50):
    w = w - 0.01 * grad
    # w = 0.5 → 0.49 → ... → 0.0 (崩溃!)

# 约束版本
log_w = log(0.5)  # ≈ -0.693
grad = 0.01
for _ in range(50):
    log_w = log_w - 0.01 * grad
    w = exp(log_w)
    # w = 0.5 → 0.495 → ... → 0.45 (仍然 > 0!)
```

---

## 🚀 使用方法

### 方法 1: 直接使用约束版本

**修改训练脚本**：

```python
# 原来的导入
# from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2

# 新的导入
from models.e3_gnn_encoder_v2_fixed import RNAPocketEncoderV2Fixed

# 创建模型（其他参数完全相同）
model = RNAPocketEncoderV2Fixed(
    num_atom_types=encoder.num_atom_types,
    num_residues=encoder.num_residues,
    hidden_irreps="32x0e + 16x1o + 8x2e",
    output_dim=512,
    num_layers=4,
    use_multi_hop=True,
    use_nonbonded=True
)
```

**就这么简单！** 其他代码无需任何修改。

---

### 方法 2: 命令行参数控制

**修改 `scripts/04_train_model.py`**：

```python
# 添加参数
parser.add_argument(
    '--use_weight_constraints',
    action='store_true',
    help='Use fixed version with weight constraints'
)

# 根据参数选择模型
if args.use_weight_constraints:
    from models.e3_gnn_encoder_v2_fixed import RNAPocketEncoderV2Fixed as ModelClass
else:
    from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2 as ModelClass

model = ModelClass(...)
```

**训练时**：
```bash
# 使用约束版本
python scripts/04_train_model.py \
    --use_weight_constraints \
    --use_multi_hop \
    --use_nonbonded

# 使用标准版本
python scripts/04_train_model.py \
    --use_multi_hop \
    --use_nonbonded
```

---

## 📊 监控权重

### 获取权重信息

```python
# 标准版本
print(f"Angle weight: {model.angle_weight.item():.4f}")

# 约束版本 - 完全相同的 API！
print(f"Angle weight: {model.angle_weight.item():.4f}")

# 约束版本 - 额外信息
summary = model.get_weight_summary()
print(summary)
# {
#     'angle_weight': 0.4823,
#     'angle_log_weight': -0.7291,  # 内部参数
#     'dihedral_weight': 0.3156,
#     'dihedral_log_weight': -1.1543,
#     'nonbonded_weight': 0.1891,
#     'nonbonded_log_weight': -1.6653
# }
```

### 训练中监控

```python
def train_epoch(model, loader, optimizer):
    for batch in loader:
        loss = compute_loss(model(batch), batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 打印权重
    if hasattr(model, 'get_weight_summary'):
        # 约束版本 - 详细信息
        print(model.get_weight_summary())
    else:
        # 标准版本 - 基本信息
        print(f"Angle: {model.angle_weight.item():.4f}")
```

---

## ⚖️ 何时使用哪个版本？

### 使用约束版本 (Fixed) 的情况：

✅ **数据不完整**
- 某些样本缺少 `triple_index` 或 `quadra_index`
- 多跳路径数量很少
- 正在生成/调试数据

✅ **训练稳定性**
- 需要保证权重不会崩溃
- 长时间训练
- 自动化实验（无人监督）

✅ **探索性实验**
- 不确定数据质量
- 快速原型开发
- 初步测试

### 使用标准版本的情况：

✅ **数据完整且高质量**
- 所有样本都有完整的多跳路径
- 已经验证过数据格式
- 权重训练稳定

✅ **性能优化**
- 标准版本理论上梯度流更直接
- 对于完美数据可能收敛稍快

✅ **基准对比**
- 需要与原始论文对比
- 标准实现

---

## 🔬 验证测试

### 测试权重约束

运行测试脚本：
```bash
python models/e3_gnn_encoder_v2_fixed.py
```

**预期输出**：
```
Initial weights:
   angle_weight: 0.500000
   dihedral_weight: 0.300000
   nonbonded_weight: 0.200000

Simulating 100 steps with strong negative gradients...
   Step 0:
     angle_weight: 0.505025
     ...

Final weights after 100 steps:
   angle_weight: 1.359140 (still > 0!)
   dihedral_weight: 0.815484 (still > 0!)
   nonbonded_weight: 0.543656 (still > 0!)

✅ Weights remain positive even with strong negative gradients!
```

### 测试模型等价性

```python
import torch
from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2
from models.e3_gnn_encoder_v2_fixed import RNAPocketEncoderV2Fixed

# 创建两个模型
model1 = RNAPocketEncoderV2(num_atom_types=71, num_residues=43)
model2 = RNAPocketEncoderV2Fixed(num_atom_types=71, num_residues=43)

# 复制权重（确保初始状态相同）
model2.load_state_dict(model1.state_dict(), strict=False)

# 测试前向传播
from torch_geometric.data import Data
data = Data(
    x=torch.randn(50, 4),
    pos=torch.randn(50, 3),
    edge_index=torch.randint(0, 50, (2, 100)),
    edge_attr=torch.randn(100, 2),
    triple_index=torch.randint(0, 50, (3, 80)),
    triple_attr=torch.randn(80, 2),
    quadra_index=torch.randint(0, 50, (4, 40)),
    quadra_attr=torch.randn(40, 3),
    nonbonded_edge_index=torch.randint(0, 50, (2, 100)),
    nonbonded_edge_attr=torch.randn(100, 3)
)

out1 = model1(data)
out2 = model2(data)

print(f"Output difference: {(out1 - out2).abs().max().item()}")
# 应该非常小 (< 1e-5)
```

---

## 📝 迁移检查清单

从标准版本迁移到约束版本：

- [ ] 修改导入语句
- [ ] 确认参数设置相同
- [ ] （可选）添加权重监控
- [ ] 运行测试验证输出一致
- [ ] 开始训练
- [ ] 监控第一个 epoch 的权重值
- [ ] 确认权重保持在合理范围 (0.1-2.0)

---

## 🛠️ 故障排查

### Q1: 约束版本训练更慢？

**A**: 不会。`exp()` 操作的开销可以忽略不计，训练速度应该相同。

### Q2: 权重值看起来不同？

**A**: 由于参数化不同，优化路径会略有不同，但最终性能应该相似。

### Q3: 如何加载旧的检查点？

**A**:
```python
# 加载标准版本的检查点到约束版本
checkpoint = torch.load('old_checkpoint.pt')

# 手动转换权重参数
if 'angle_weight' in checkpoint['model_state_dict']:
    w = checkpoint['model_state_dict']['angle_weight']
    checkpoint['model_state_dict']['angle_log_weight'] = w.log()
    del checkpoint['model_state_dict']['angle_weight']

# 同样处理 dihedral_weight 和 nonbonded_weight
# ...

model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

### Q4: 权重变得很大 (> 10)？

**A**: 这可能表明模型过度依赖某个路径。考虑：
- 检查数据是否不平衡
- 添加权重正则化
- 检查其他路径是否有问题

---

## 💡 最佳实践

1. **开发阶段**: 始终使用约束版本
   - 数据可能不完美
   - 需要快速迭代
   - 避免意外崩溃

2. **生产训练**: 根据数据质量选择
   - 数据完整 → 标准版本
   - 数据不确定 → 约束版本

3. **监控**: 无论使用哪个版本，都要监控权重
   - 第一个 epoch 后检查
   - 每 10 个 epoch 记录
   - 出现异常立即检查

4. **实验对比**: 两个版本都试一下
   - 约束版本应该更稳定
   - 标准版本在完美数据上可能略好
   - 记录差异

---

## 🔗 相关文档

- `FIX_ZERO_WEIGHTS.md`: 权重归零问题的完整诊断
- `MODELS_V2_SUMMARY.md`: v2.0 模型架构概述
- `TRAINING_GUIDE_V2.md`: v2.0 训练完整指南

---

**总结**：
- ⭐ **约束版本是更安全的选择**
- ⭐ **API 完全兼容，迁移无痛**
- ⭐ **数学保证权重 > 0**
- ⭐ **适合大多数使用场景**
