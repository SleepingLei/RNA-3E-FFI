# 梯度爆炸诊断 - 快速开始

## 🚀 一键诊断

```bash
python scripts/diagnose_gradient_explosion.py \
    --model_version v3 \
    --data_dir data/processed_pockets \
    --ligand_embeddings data/ligand_embeddings.h5 \
    --split_dir data/splits \
    --epochs 10 \
    --batch_size 32 \
    --output_dir gradient_diagnosis
```

**等待输出，如果梯度爆炸会自动停止并生成报告。**

---

## 📊 查看结果

### 1. 快速查看图表

```bash
# Mac
open gradient_diagnosis/diagnostics.png
open gradient_diagnosis/gradient_heatmap.png
open gradient_diagnosis/feature_changes.png

# Linux
xdg-open gradient_diagnosis/diagnostics.png
xdg-open gradient_diagnosis/gradient_heatmap.png
xdg-open gradient_diagnosis/feature_changes.png
```

### 2. 查看CSV数据

```bash
# 查看前20行
head -20 gradient_diagnosis/gradient_stats.csv

# 或用Excel/Numbers打开
```

### 3. 查看JSON报告

```bash
cat gradient_diagnosis/diagnosis_report.json | python -m json.tool | less
```

---

## 🔍 典型输出

### 正常情况

```
Epoch 1/10
  Batch 0/100: Loss=0.5234, GradNorm=2.3456
  Batch 10/100: Loss=0.4982, GradNorm=2.1234
  Batch 20/100: Loss=0.4756, GradNorm=1.9876
  ...

Epoch 1 Summary:
  Train Loss: 0.4523
  Val Loss: 0.4789
  ✓ Best model saved
```

### 检测到梯度爆炸

```
  Batch 45/100: Loss=15.6789, GradNorm=1523.4567

⚠️  Step 145 - 检测到异常!
  梯度异常层: [...angle_mp_layers.1...]
  Total grad norm: 1523.4567

❌ 检测到梯度爆炸!
  停止训练并生成报告...

✓ 报告已保存到: gradient_diagnosis/
```

---

## 📁 输出文件

| 文件 | 内容 | 用途 |
|------|------|------|
| `gradient_stats.csv` | 每步的梯度、loss、学习率 | Excel分析 |
| `input_features.csv` | **NEW** 每步的输入特征统计 | 检测数据异常 |
| `diagnosis_report.json` | 综合报告 | 程序化分析 |
| `diagnostics.png` | 4合1图表 | 快速查看趋势 |
| `gradient_heatmap.png` | 梯度热图 | 定位问题层 |
| `feature_changes.png` | **NEW** 特征变化图表 | 检测输入异常 |
| `best_model.pt` | 最佳模型 | （如果训练完成）|

---

## 🐛 问题诊断

### Step 1: 查看热图

**打开** `gradient_heatmap.png`

```
问：哪些层是红色的？
答：angle_mp_layers.0, angle_mp_layers.1

问：从哪个step开始变红？
答：Step 120 左右
```

### Step 2: 查看CSV

```python
import pandas as pd
df = pd.read_csv('gradient_diagnosis/gradient_stats.csv')

# 找出Step 120附近发生了什么
print(df[df['step'].between(110, 130)])
```

### Step 3: 查看JSON报告

```python
import json
with open('gradient_diagnosis/diagnosis_report.json') as f:
    report = json.load(f)

# 看看权重变化最大的层
print(report['weight_changes'][:5])
```

---

## 🔧 常见问题和解决方案

### 问题 1: angle_mp_layers 梯度爆炸

**症状**:
- 热图显示 `angle_mp_layers.*` 是红色
- CSV显示这些层的 grad_norm > 10

**原因**:
- 可能是 `angle_deviation` 导致的（已在简化版V3中移除）
- 或 LayerNorm 缺失

**解决**:
```bash
# 确认使用的是简化版V3
git pull  # 确保代码是最新的

# 降低学习率
--lr 5e-5

# 更严格的梯度裁剪
--grad_clip 1.0
```

---

### 问题 2: 训练初期就不稳定

**症状**:
- 第一个epoch内梯度就 > 100
- Loss 突然跳到很大

**原因**:
- 学习率过大
- 数据归一化问题

**解决**:
```bash
# 大幅降低学习率
--lr 1e-5

# 检查数据
python scripts/check_data_normalization.py

# 更小的batch size
--batch_size 16
```

---

### 问题 3: 某个epoch突然爆炸

**症状**:
- Epoch 1-3 正常
- Epoch 4 突然爆炸

**原因**:
- 数据集中某些异常样本
- 学习率scheduler问题

**解决**:
```bash
# 查看是哪个样本引起的
# 在代码中添加
print(f"Processing complex_id: {batch.complex_id}")

# 固定学习率，不用scheduler
# 修改代码注释掉: scheduler.step()

# 检查数据质量
python scripts/check_data_outliers.py
```

---

### 问题 4: 输入特征异常 *NEW*

**症状**:
- `input_features.csv` 显示特征值突变
- `feature_changes.png` 中有红色异常点
- 警告信息显示 "输入特征异常"

**原因**:
- 数据预处理错误（归一化问题）
- 某些样本包含异常值
- 数据加载器出错

**解决**:
```python
# 1. 查看哪些特征异常
import pandas as pd
df = pd.read_csv('gradient_diagnosis/input_features.csv')
anomalies = df[df['suspicious_features'] != '']
print(anomalies)

# 2. 检查特征范围
print(f"Charge range: [{df['x_col0_min'].min():.2f}, {df['x_col0_max'].max():.2f}]")
print(f"Pos std range: [{df['pos_std'].min():.2f}, {df['pos_std'].max():.2f}]")

# 3. 找出发生异常的step
print(f"First anomaly at step: {anomalies['step'].min()}")
```

**修复**:
```bash
# 检查数据预处理
python scripts/01_process_data.py --check_only

# 重新处理数据
python scripts/01_process_data.py --recalculate_stats
```

---

## 📈 分析脚本模板

### 快速分析

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('gradient_diagnosis/gradient_stats.csv')

# 关键统计
print(f"最大梯度范数: {df['total_grad_norm'].max()}")
print(f"平均梯度范数: {df['total_grad_norm'].mean()}")
print(f"梯度爆炸次数 (>100): {(df['total_grad_norm'] > 100).sum()}")

# 可视化
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

ax[0].plot(df['step'], df['loss'])
ax[0].set_ylabel('Loss')
ax[0].set_yscale('log')
ax[0].set_title('Loss Curve')

ax[1].plot(df['step'], df['total_grad_norm'])
ax[1].axhline(y=10, color='r', linestyle='--', label='Warning')
ax[1].axhline(y=100, color='red', linestyle='--', label='Explosion')
ax[1].set_ylabel('Gradient Norm')
ax[1].set_xlabel('Step')
ax[1].set_yscale('log')
ax[1].legend()
ax[1].set_title('Gradient Norm')

plt.tight_layout()
plt.savefig('my_analysis.png', dpi=150)
plt.show()
```

---

## 💡 最佳实践

### 1. 首次训练新模型

```bash
# 先用诊断脚本跑5-10个epoch
python scripts/diagnose_gradient_explosion.py \
    --epochs 10 \
    --output_dir initial_diagnosis

# 如果稳定，再用正式脚本训练
python scripts/04_train_model.py \
    --epochs 100
```

### 2. 修改模型后

```bash
# 每次修改模型都先诊断
python scripts/diagnose_gradient_explosion.py \
    --epochs 5 \
    --output_dir after_modification_diagnosis
```

### 3. 对比不同配置

```bash
# 配置A
python scripts/diagnose_gradient_explosion.py \
    --lr 1e-4 --grad_clip 1.5 \
    --output_dir config_A

# 配置B
python scripts/diagnose_gradient_explosion.py \
    --lr 5e-5 --grad_clip 1.0 \
    --output_dir config_B

# 对比结果
python compare_configs.py config_A config_B
```

---

## 🎯 关键指标

| 指标 | 正常范围 | 警告 | 危险 |
|------|---------|------|------|
| total_grad_norm | 0.1 - 5.0 | 5.0 - 10.0 | > 10.0 |
| max_grad_norm | 0.5 - 3.0 | 3.0 - 10.0 | > 10.0 |
| loss (cosine) | 0.2 - 0.8 | > 1.0 | > 5.0 |
| num_nan_grads | 0 | 0 | > 0 |
| num_inf_grads | 0 | 0 | > 0 |

---

## 🆘 紧急情况

### 梯度爆炸了怎么办？

1. **不要慌**，诊断脚本已经保存了所有信息

2. **查看热图**，定位问题层
   ```bash
   open gradient_diagnosis/gradient_heatmap.png
   ```

3. **查看CSV**，找到爆炸的step
   ```python
   df = pd.read_csv('gradient_diagnosis/gradient_stats.csv')
   explosion_step = df[df['total_grad_norm'] > 100].iloc[0]
   print(explosion_step)
   ```

4. **应用临时修复**
   ```bash
   # 立即尝试这些参数
   python scripts/04_train_model.py \
       --lr 1e-5 \           # 降低学习率10倍
       --grad_clip 0.5 \     # 严格裁剪
       --batch_size 16 \     # 减小batch
       --use_amp             # 使用混合精度
   ```

5. **查看文档**
   ```bash
   cat docs/gradient_instability_diagnosis.md
   cat docs/gradient_stability_fixes.md
   ```

6. **寻求帮助**
   - 将 `gradient_diagnosis/` 目录打包
   - 提供给开发者分析

---

## 📚 相关文档

- **完整使用指南**: `docs/gradient_explosion_diagnosis_guide.md`
- **梯度问题诊断**: `docs/gradient_instability_diagnosis.md`
- **修复方案**: `docs/gradient_stability_fixes.md`
- **已应用修复**: `docs/applied_fixes_summary.md`

---

## ⚡ TL;DR

```bash
# 1. 运行诊断
python scripts/diagnose_gradient_explosion.py \
    --model_version v3 \
    --data_dir data/processed_pockets \
    --ligand_embeddings data/ligand_embeddings.h5 \
    --split_dir data/splits \
    --epochs 10 \
    --output_dir gradient_diagnosis

# 2. 查看结果
open gradient_diagnosis/diagnostics.png
open gradient_diagnosis/gradient_heatmap.png

# 3. 如果有问题，应用修复
python scripts/04_train_model.py \
    --lr 5e-5 \
    --grad_clip 1.0 \
    --batch_size 16
```

**就这么简单！** 🎉

---

*快速参考版本: 1.0*
*创建时间: 2025-11-09*
