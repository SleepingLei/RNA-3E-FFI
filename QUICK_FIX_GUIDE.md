# 🚀 快速修复指南：权重归零问题

## 📝 问题总结

你遇到的训练输出：
```
Train Loss: 199.524375
  Angle weight: 0.0000      ❌ 变成0了
  Dihedral weight: 0.0000   ❌ 变成0了
  Nonbonded weight: 0.0771
```

**根本原因**：使用了 v1.0 格式数据训练 v2.0 模型

---

## ⚡ 立即修复（2种方案）

### 方案 A：重新生成数据（推荐）⭐⭐⭐

**适用于**：想要使用完整的 v2.0 功能

**步骤**：

1. **备份旧数据**
   ```bash
   mv data/processed/graphs data/processed/graphs_v1_backup
   mkdir -p data/processed/graphs
   ```

2. **重新生成 v2.0 格式数据**
   ```bash
   python scripts/03_build_dataset.py \
       --hariboss_csv hariboss/Complexes.csv \
       --amber_dir data/processed/amber \
       --output_dir data/processed/graphs \
       --distance_cutoff 5.0 \
       --num_workers 8
   ```

3. **验证数据格式**
   ```bash
   python diagnose_weights.py
   ```

   应该看到：
   ```
   ✅ 数据格式正确，包含多跳路径
   ```

4. **重新训练（使用约束版本，更安全）**
   ```bash
   python scripts/04_train_model.py \
       --graph_dir data/processed/graphs \
       --embeddings_path data/processed/ligand_embeddings.h5 \
       --output_dir models/checkpoints_v2_fixed \
       --batch_size 4 \
       --num_epochs 100 \
       --use_multi_hop \
       --use_nonbonded \
       --use_weight_constraints
   ```

   **注意**：加了 `--use_weight_constraints` 标志！

5. **检查第一个 epoch**

   应该看到：
   ```
   Train Loss: 0.5~2.0  ✅ 正常范围
     Angle weight: 0.4~0.6     ✅ 有值
     Dihedral weight: 0.2~0.4  ✅ 有值
     Nonbonded weight: 0.1~0.3 ✅ 有值
   ```

---

### 方案 B：使用权重约束（应急）⚡

**适用于**：暂时无法重新生成数据，或数据正在生成中

**步骤**：

只需在训练时添加 `--use_weight_constraints` 标志：

```bash
python scripts/04_train_model.py \
    --graph_dir data/processed/graphs \
    --embeddings_path data/processed/ligand_embeddings.h5 \
    --output_dir models/checkpoints_constrained \
    --use_multi_hop \
    --use_nonbonded \
    --use_weight_constraints    # ← 添加这个！
```

**效果**：
- ✅ 权重永远不会变成 0
- ✅ 使用对数空间参数化 (weight = exp(log_weight))
- ⚠️  但如果数据缺失多跳路径，性能仍然受限

**优点**：
- 立即可用，无需等待数据生成
- 防止权重崩溃
- 作为临时解决方案

**缺点**：
- 如果数据不包含多跳路径，多跳功能仍然无效
- 只是"防守"措施，不解决数据问题

---

## 🔍 如何选择？

| 场景 | 推荐方案 |
|-----|---------|
| 数据肯定是 v1.0 格式 | **方案 A**（重新生成） |
| 不确定数据格式 | 先运行 `diagnose_weights.py` 检查 |
| 数据正在生成中 | **方案 B**（临时使用约束） |
| 想要最佳性能 | **方案 A**（完整数据 + 约束） |
| 快速测试 | **方案 B**（约束版本） |

---

## 📊 两种方案对比

| 项目 | 重新生成数据 | 使用权重约束 |
|-----|------------|-------------|
| 解决数据问题 | ✅ 是 | ❌ 否 |
| 防止权重归零 | ✅ 是 | ✅ 是 |
| 需要时间 | ⏱️ 10-60分钟 | ⚡ 立即 |
| 多跳功能 | ✅ 完整 | ⚠️ 取决于数据 |
| 推荐度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 🎯 推荐组合：两者都用！

**最佳实践**：
```bash
# 1. 重新生成数据
python scripts/03_build_dataset.py ...

# 2. 验证数据
python diagnose_weights.py

# 3. 使用约束版本训练（双重保险）
python scripts/04_train_model.py \
    --use_multi_hop \
    --use_nonbonded \
    --use_weight_constraints    # 即使数据正确，也加约束作为安全措施
```

**为什么**：
- ✅ 数据完整 → 多跳功能正常工作
- ✅ 权重约束 → 即使出现边缘情况，权重也不会崩溃
- ✅ 稳定训练 → 可以安心长时间训练

---

## 🔧 故障排查

### Q1: 重新生成数据后仍然权重归零？

**检查**：
```bash
python diagnose_weights.py
```

**可能原因**：
1. 数据生成时出错，检查日志
2. `triple_index` 或 `quadra_index` 为空（数量为 0）
3. 使用了错误的图文件目录

**解决**：
```bash
# 检查单个文件
python -c "
import torch
data = torch.load('data/processed/graphs/xxx.pt')
print(f'Features: {data.x.shape}')
print(f'Angles: {data.triple_index.shape if hasattr(data, 'triple_index') else 'MISSING'}')
"
```

### Q2: 使用约束版本后权重变得很大（> 10）？

**正常！** 这可能表示模型过度依赖某个路径。

**检查**：
- 其他路径是否正常工作？
- 损失是否下降？

**如果担心**：
```python
# 可以添加权重正则化（需要修改训练脚本）
# 在损失函数中添加：
weight_reg = (model.angle_weight - 1.0)**2 + \
             (model.dihedral_weight - 1.0)**2 + \
             (model.nonbonded_weight - 1.0)**2
loss = mse_loss + 0.01 * weight_reg
```

### Q3: 训练脚本找不到约束版本模型？

**错误**：
```
ImportError: cannot import name 'RNAPocketEncoderV2Fixed'
```

**检查**：
```bash
# 确认文件存在
ls models/e3_gnn_encoder_v2_fixed.py

# 如果不存在，说明文件未创建
# 重新运行之前创建固定版本的步骤
```

---

## 📈 预期训练输出

### 标准版本（数据完整）
```
Using RNAPocketEncoderV2 (standard)
...
Epoch 1:
  Train Loss: 1.234567
  Angle weight: 0.4892      ✅ 正常
  Dihedral weight: 0.2956   ✅ 正常
  Nonbonded weight: 0.1832  ✅ 正常

Epoch 10:
  Train Loss: 0.567890
  Angle weight: 0.5234      ✅ 缓慢变化
  Dihedral weight: 0.3102   ✅ 缓慢变化
  Nonbonded weight: 0.1654  ✅ 缓慢变化
```

### 约束版本（数据完整）
```
Using RNAPocketEncoderV2Fixed (with weight constraints)
...
Epoch 1:
  Train Loss: 1.234567
  Angle weight: 0.4892      ✅ 正常
  Dihedral weight: 0.2956   ✅ 正常
  Nonbonded weight: 0.1832  ✅ 正常

Epoch 10:
  Train Loss: 0.567890
  Angle weight: 0.5234      ✅ 永远 > 0
  Dihedral weight: 0.3102   ✅ 永远 > 0
  Nonbonded weight: 0.1654  ✅ 永远 > 0

Weight constraints (log-space parameters):
  angle_log_weight: -0.6471
  dihedral_log_weight: -1.1712
  nonbonded_log_weight: -1.7982
```

### 错误情况（数据缺失 + 无约束）
```
Using RNAPocketEncoderV2 (standard)
...
Epoch 1:
  Train Loss: 150.234567    ⚠️ 损失很高
  Angle weight: 0.4998      ⚠️ 开始下降
  Dihedral weight: 0.2999
  Nonbonded weight: 0.1999

Epoch 10:
  Train Loss: 199.567890    ❌ 损失不降
  Angle weight: 0.0000      ❌ 归零了！
  Dihedral weight: 0.0000   ❌ 归零了！
  Nonbonded weight: 0.0771
```

---

## 📞 需要更多帮助？

1. **查看详细文档**：
   - `FIX_ZERO_WEIGHTS.md`: 问题诊断详解
   - `WEIGHT_CONSTRAINT_GUIDE.md`: 约束版本使用指南
   - `TRAINING_GUIDE_V2.md`: 完整训练指南

2. **运行诊断工具**：
   ```bash
   python diagnose_weights.py
   ```

3. **检查数据格式**：
   ```bash
   python scripts/test_v2_features.py
   ```

---

## ✅ 成功检查清单

训练成功的标志：

- [ ] 第一个 epoch 后权重都 > 0
- [ ] 训练损失稳定下降
- [ ] 权重在合理范围 (0.1-2.0)
- [ ] 验证损失随时间下降
- [ ] 没有 NaN 或 Inf
- [ ] GPU/CPU 利用率正常

如果全部勾选 → 🎉 训练正常！

---

**最后提醒**：
- ⚡ 方案 B（约束）可以立即使用
- ⭐ 方案 A（重新生成）是长期解决方案
- 🎯 推荐两者结合使用
- 📊 训练时持续监控权重值
