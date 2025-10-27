# 模型评估指南

## 测试集评估脚本

`scripts/evaluate_test_set.py` 脚本用于在测试集上评估训练好的模型，并计算不同阈值下的命中率（Hit Rate）。

## 功能特性

### 评估指标

- **Hit Rate @ top-k%**: 正确配体是否出现在检索结果的前k%中
  - 默认计算 top-5%, top-10%, top-20% 的命中率
  - 可自定义阈值百分比

- **排名统计**: 正确配体在所有候选中的排名
  - 平均排名 (Mean Rank)
  - 中位数排名 (Median Rank)
  - 最小/最大排名
  - 标准差

- **距离统计**: 预测embedding与正确配体embedding之间的距离
  - 平均距离、中位数距离
  - 最小/最大距离
  - 标准差

### 支持的距离度量

- **Cosine Distance** (推荐，默认): `1 - cosine_similarity`
- **Euclidean Distance**: L2距离

## 使用方法

### 基本用法

```bash
python scripts/evaluate_test_set.py \
    --checkpoint models/checkpoints/best_model.pt \
    --splits data/splits/splits.json \
    --graph_dir data/processed/graphs \
    --ligand_embeddings data/processed/ligand_embeddings.h5
```

### 完整参数

```bash
python scripts/evaluate_test_set.py \
    --checkpoint models/checkpoints/best_model.pt \
    --splits data/splits/splits.json \
    --graph_dir data/processed/graphs \
    --ligand_embeddings data/processed/ligand_embeddings.h5 \
    --output results/test_evaluation.json \
    --metric cosine \
    --top_percentages 5 10 20
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | 是 | - | 训练好的模型checkpoint路径 |
| `--splits` | 否 | `data/splits/splits.json` | 数据集划分文件 |
| `--graph_dir` | 否 | `data/processed/graphs` | 图数据文件目录 |
| `--ligand_embeddings` | 否 | `data/processed/ligand_embeddings.h5` | 配体embeddings文件 |
| `--output` | 否 | None | 结果保存路径（JSON格式） |
| `--metric` | 否 | `cosine` | 距离度量方式 (`cosine`/`euclidean`) |
| `--top_percentages` | 否 | `5 10 20` | 命中率计算阈值（百分比） |

## 输出结果

### 终端输出示例

```
============================================================
Test Set Evaluation Results
============================================================

Total samples:           95
Successful predictions:  92
Failed predictions:      3
Total ligands in library: 856
Distance metric:         cosine

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
  Std rank:    78.54

============================================================
Distance Statistics (to correct ligand)
============================================================
  Mean distance:   0.234567
  Median distance: 0.189012
  Min distance:    0.012345
  Max distance:    0.789012
  Std distance:    0.145678
```

### JSON输出格式

保存的JSON文件包含详细的评估结果：

```json
{
  "total_samples": 95,
  "successful_predictions": 92,
  "failed_predictions": [
    {
      "complex_id": "1abc_LIG_model0",
      "reason": "graph_file_not_found"
    }
  ],
  "metric": "cosine",
  "total_ligands": 856,
  "hit_rates": {
    "top5%": {
      "hits": 42,
      "misses": 50,
      "hit_rate": 45.65,
      "k_value": 43
    },
    "top10%": {
      "hits": 63,
      "misses": 29,
      "hit_rate": 68.48,
      "k_value": 86
    },
    "top20%": {
      "hits": 78,
      "misses": 14,
      "hit_rate": 84.78,
      "k_value": 171
    }
  },
  "rank_statistics": {
    "mean_rank": 65.32,
    "median_rank": 45.0,
    "min_rank": 1,
    "max_rank": 512,
    "std_rank": 78.54
  },
  "distance_statistics": {
    "mean_distance": 0.234567,
    "median_distance": 0.189012,
    "min_distance": 0.012345,
    "max_distance": 0.789012,
    "std_distance": 0.145678
  },
  "detailed_results": [
    {
      "complex_id": "1aju_ARG_model1",
      "true_ligand_id": "1aju_ARG",
      "rank": 3,
      "distance": 0.123456,
      "hits": {
        "top5%": true,
        "top10%": true,
        "top20%": true
      }
    }
  ]
}
```

## 评估流程

1. **加载模型**: 从checkpoint自动检测模型版本和配置
2. **加载测试集**: 从`splits.json`读取test部分的样本ID
3. **加载配体库**: 加载所有配体的embeddings
4. **推理预测**: 对每个测试样本预测pocket embedding
5. **检索排序**: 计算与所有配体的距离并排序
6. **计算指标**: 统计正确配体的排名和命中率

## 命中率计算说明

### Top-k% Hit Rate

对于每个测试样本：
1. 预测pocket的embedding
2. 计算与所有候选配体的距离
3. 按距离排序（距离越小越相似）
4. 检查正确配体是否在前k%的候选中

示例：
- 如果配体库有1000个配体
- top-5%意味着前50个候选（1000 × 5% = 50）
- 如果正确配体排名≤50，则算作命中

### 排名 (Rank)

正确配体在所有候选中的位置（1-indexed）：
- Rank = 1: 模型预测完全正确（最相似）
- Rank = 10: 正确配体是第10相似的
- Mean Rank越小越好，理想情况接近1

## 常见问题

### Q1: 为什么有failed predictions？

**A**: 可能的原因：
- 图文件不存在或损坏
- 配体embedding不在库中
- 图加载或预测过程出错

查看输出中的失败原因统计。

### Q2: 如何提高命中率？

**A**: 可能的改进方向：
- 增加训练数据
- 调整模型架构（更多layers、更大hidden_dim）
- 尝试不同的距离度量
- 改进特征工程
- 调整训练超参数

### Q3: Cosine vs Euclidean距离？

**A**:
- **Cosine**: 关注方向相似性，对embedding的尺度不敏感（推荐）
- **Euclidean**: 关注绝对距离，受embedding尺度影响

通常cosine距离在embedding检索任务中表现更好。

### Q4: 结果中的model number是什么？

**A**:
- 数据处理时为每个complex生成了多个构象（models）
- 例如：`1aju_ARG_model0`, `1aju_ARG_model1`, `1aju_ARG_model2`
- 它们共享同一个配体ID：`1aju_ARG`
- 测试时每个model独立评估

### Q5: 如何解读排名统计？

**A**:
- **Mean/Median Rank**: 越小越好，理想接近1
- **Min Rank = 1**: 至少有一个样本预测完全正确
- **Max Rank很大**: 说明有一些difficult cases
- **Std很大**: 说明模型在不同样本上的表现差异大

## 使用示例

### 示例1: 快速评估

```bash
# 使用默认参数评估模型
python scripts/evaluate_test_set.py \
    --checkpoint models/checkpoints/best_model.pt
```

### 示例2: 自定义阈值

```bash
# 评估top-1%, 5%, 10%, 15%, 20%的命中率
python scripts/evaluate_test_set.py \
    --checkpoint models/checkpoints/best_model.pt \
    --top_percentages 1 5 10 15 20 \
    --output results/detailed_evaluation.json
```

### 示例3: 使用Euclidean距离

```bash
# 使用L2距离而不是cosine距离
python scripts/evaluate_test_set.py \
    --checkpoint models/checkpoints/best_model.pt \
    --metric euclidean \
    --output results/eval_euclidean.json
```

### 示例4: 比较多个checkpoint

```bash
# 创建脚本批量评估多个checkpoint
for ckpt in models/checkpoints/epoch_*.pt; do
    echo "Evaluating $ckpt"
    python scripts/evaluate_test_set.py \
        --checkpoint "$ckpt" \
        --output "results/$(basename $ckpt .pt)_eval.json"
done
```

## 输出文件

评估脚本会生成：
- **终端输出**: 汇总统计和命中率
- **JSON文件** (可选): 包含完整的评估结果和每个样本的详细信息

建议保存JSON文件以便后续分析和可视化。

## 后续分析

可以使用JSON结果进行进一步分析：

```python
import json
import matplotlib.pyplot as plt
import numpy as np

# 加载结果
with open('results/test_evaluation.json', 'r') as f:
    results = json.load(f)

# 绘制rank分布
ranks = [r['rank'] for r in results['detailed_results']]
plt.hist(ranks, bins=50)
plt.xlabel('Rank')
plt.ylabel('Count')
plt.title('Distribution of Correct Ligand Ranks')
plt.savefig('rank_distribution.png')

# 分析困难样本
difficult_samples = [r for r in results['detailed_results'] if r['rank'] > 100]
print(f"Difficult samples (rank > 100): {len(difficult_samples)}")
```

## 注意事项

1. **数据一致性**: 确保使用的图和配体embeddings与训练时一致
2. **标准化**: 如果训练时使用了标准化，测试时也需要应用相同的标准化参数
3. **设备选择**: 自动使用GPU（如果可用），否则使用CPU
4. **内存管理**: 配体库较大时会占用较多内存

## 相关文件

- 训练脚本: `scripts/04_train_model.py`
- 推理脚本: `scripts/05_run_inference.py`
- 数据划分: `data/splits/splits.json`
- 标准化指南: `NORMALIZATION_GUIDE.md`

---

**问题反馈**: 如有问题，请查看代码注释或提交issue
