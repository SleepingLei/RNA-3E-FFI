# 模型评估 - 快速参考

## 🎯 快速开始

### 评估测试集

```bash
# 基本用法
python scripts/evaluate_test_set.py \
    --checkpoint models/checkpoints/best_model.pt

# 完整参数
python scripts/evaluate_test_set.py \
    --checkpoint models/checkpoints/best_model.pt \
    --splits data/splits/splits.json \
    --graph_dir data/processed/graphs \
    --ligand_embeddings data/processed/ligand_embeddings.h5 \
    --output results/test_evaluation.json \
    --metric cosine \
    --top_percentages 5 10 20
```

### 运行示例脚本

```bash
# 运行预配置的评估示例
./run_evaluation_example.sh
```

## 📊 评估指标

| 指标 | 说明 |
|------|------|
| **Hit Rate @ top-k%** | 正确配体是否在前k%候选中 |
| **Mean Rank** | 正确配体的平均排名（越小越好） |
| **Median Rank** | 正确配体的中位数排名 |
| **Distance Statistics** | 预测与正确配体的距离统计 |

### 命中率计算

- **top-5%**: 正确配体在前5%候选中的比例
- **top-10%**: 正确配体在前10%候选中的比例
- **top-20%**: 正确配体在前20%候选中的比例

**示例**:
- 配体库有1000个配体
- top-5% = 前50个候选
- 如果正确配体排名≤50，算作命中

## 📁 输入文件

| 文件 | 路径 | 说明 |
|------|------|------|
| **Checkpoint** | `models/checkpoints/best_model.pt` | 训练好的模型 |
| **Splits** | `data/splits/splits.json` | 数据集划分（train/val/test） |
| **Graphs** | `data/processed/graphs/*.pt` | RNA pocket图数据 |
| **Ligand Embeddings** | `data/processed/ligand_embeddings.h5` | 配体embeddings库 |

## 📤 输出结果

### 终端输出

```
============================================================
Test Set Evaluation Results
============================================================

Total samples:           95
Successful predictions:  92
Failed predictions:      3

============================================================
Hit Rates
============================================================

  top5% (k=43):
  Hit rate:  45.65% (42/92)

 top10% (k=86):
  Hit rate:  68.48% (63/92)

 top20% (k=171):
  Hit rate:  84.78% (78/92)

============================================================
Rank Statistics (of correct ligand)
============================================================
  Mean rank:   65.32
  Median rank: 45.0
  Min rank:    1
  Max rank:    512
```

### JSON输出

保存详细结果到JSON文件（使用`--output`参数）：
- 每个样本的排名和距离
- 命中率详细统计
- 失败样本信息

## 🛠️ 常用命令

### 1. 基础评估
```bash
python scripts/evaluate_test_set.py \
    --checkpoint models/checkpoints/best_model.pt \
    --output results/eval.json
```

### 2. 自定义阈值
```bash
python scripts/evaluate_test_set.py \
    --checkpoint models/checkpoints/best_model.pt \
    --top_percentages 1 5 10 15 20 25 30
```

### 3. 使用Euclidean距离
```bash
python scripts/evaluate_test_set.py \
    --checkpoint models/checkpoints/best_model.pt \
    --metric euclidean
```

### 4. 批量评估多个checkpoint
```bash
for ckpt in models/checkpoints/epoch_*.pt; do
    python scripts/evaluate_test_set.py \
        --checkpoint "$ckpt" \
        --output "results/$(basename $ckpt .pt)_eval.json"
done
```

## 🔍 结果分析

### 查看JSON结果

```python
import json

# 加载结果
with open('results/test_evaluation.json', 'r') as f:
    results = json.load(f)

# 查看命中率
print(results['hit_rates'])

# 查看排名统计
print(results['rank_statistics'])

# 查看困难样本（排名>100）
difficult = [r for r in results['detailed_results'] if r['rank'] > 100]
print(f"Difficult samples: {len(difficult)}")
```

### 可视化

```python
import matplotlib.pyplot as plt
import numpy as np

# 排名分布
ranks = [r['rank'] for r in results['detailed_results']]
plt.hist(ranks, bins=50)
plt.xlabel('Rank')
plt.ylabel('Count')
plt.title('Distribution of Correct Ligand Ranks')
plt.savefig('rank_distribution.png')
```

## 📋 评估流程

1. **加载模型** → 自动检测版本和配置
2. **加载测试集** → 从splits.json读取test部分
3. **加载配体库** → 加载所有配体embeddings
4. **推理预测** → 对每个测试样本预测embedding
5. **检索排序** → 计算距离并排序
6. **计算指标** → 统计排名和命中率

## ⚠️ 注意事项

1. **数据一致性**: 确保图数据和embeddings与训练时一致
2. **标准化**: 如果训练时使用了标准化，确保测试数据也已标准化
3. **模型版本**: 脚本自动检测模型版本（v1/v2）
4. **内存**: 配体库较大时会占用较多内存

## 💡 常见问题

**Q: 为什么有failed predictions？**
A: 检查输出中的失败原因，可能是文件缺失或数据格式问题

**Q: Cosine vs Euclidean距离？**
A: Cosine距离通常在embedding检索任务中表现更好（推荐使用）

**Q: 如何提高命中率？**
A: 增加训练数据、调整模型架构、改进特征工程、优化超参数

**Q: Rank=1是什么意思？**
A: 表示模型完美预测，正确配体是最相似的候选

## 📚 详细文档

完整的评估指南请参考: [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)

## 🔗 相关文件

- **评估脚本**: `scripts/evaluate_test_set.py`
- **训练脚本**: `scripts/04_train_model.py`
- **推理脚本**: `scripts/05_run_inference.py`
- **数据划分**: `data/splits/splits.json`
- **示例脚本**: `run_evaluation_example.sh`

---

**快速帮助**: `python scripts/evaluate_test_set.py --help`
