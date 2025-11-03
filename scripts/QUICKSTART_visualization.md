# 快速入门：Embedding 可视化分析

## 5 分钟快速开始

### 方法 1: 一键运行（推荐）

```bash
# 使用默认参数运行完整分析流程
bash scripts/run_embedding_analysis.sh
```

### 方法 2: 分析特定数据集（train/val/test）

```bash
# 只分析测试集
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits test

# 分析验证集 + 测试集
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits "val test"

# 分析所有数据（train + val + test）
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits "train val test"

# 只分析训练集
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits train
```

### 方法 3: 自定义参数

```bash
# 指定自定义路径
bash scripts/run_embedding_analysis.sh \
    --checkpoint models/checkpoints/your_model.pt \
    --graph_dir data/processed/your_graphs \
    --output_dir results/your_analysis
```

### 方法 4: 分步运行

```bash
# 步骤 1: 主可视化（所有数据）
python scripts/visualize_embeddings.py \
    --checkpoint models/checkpoints/best_model.pt \
    --graph_dir data/processed/graphs \
    --ligand_embeddings data/processed/ligand_embeddings_dedup.h5 \
    --output_dir results/visualizations

# 步骤 1b: 主可视化（仅测试集）
python scripts/visualize_embeddings.py \
    --checkpoint models/checkpoints/best_model.pt \
    --graph_dir data/processed/graphs \
    --ligand_embeddings data/processed/ligand_embeddings_dedup.h5 \
    --output_dir results/visualizations_test \
    --splits_file data/splits/splits.json \
    --splits test

# 步骤 2: 高级分析
python scripts/advanced_embedding_analysis.py \
    --matched_pairs results/visualizations/matched_pairs.json \
    --output_dir results/advanced_analysis
```

---

## 数据集分组功能（Train/Val/Test Splits）

### 功能说明

可以选择分析特定的数据集分组，这对于以下场景非常有用：
- **模型评估**：只在测试集上评估，避免训练集污染
- **过拟合检测**：比较训练集和测试集的性能差异
- **快速迭代**：在较小的验证集上快速测试
- **泛化能力**：评估模型在未见数据上的表现

### 使用方法

需要同时指定两个参数：
1. `--splits_file`: splits.json 文件的路径
2. `--splits`: 要分析的数据集（train, val, test 或它们的组合）

### 实际例子

```bash
# 例子 1: 只分析测试集（最常用，评估最终性能）
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits test

# 例子 2: 分析验证集和测试集（评估泛化能力）
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits "val test"

# 例子 3: 分析所有数据
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits "train val test"

# 例子 4: 环境变量方式
export SPLITS_FILE=data/splits/splits.json
export SPLITS="test"
bash scripts/run_embedding_analysis.sh
```

### splits.json 文件格式

```json
{
  "train": [
    "2m4q_AM2_model1",
    "5wnv_B6M_model0",
    ...
  ],
  "val": [
    "1aju_ARG_model0",
    ...
  ],
  "test": [
    "2kx8_GTP_model1",
    ...
  ]
}
```

每个分组包含 complex ID 列表（与 graph 文件名对应，不含 .pt 扩展名）。

### 注意事项

1. **文件名匹配**：确保 splits.json 中的 ID 与 graph 文件名匹配
   ```bash
   # splits.json 中: "1aju_ARG_model0"
   # graph 文件名: 1aju_ARG_model0.pt  ✓ 正确
   ```

2. **区分大小写**：ID 匹配是大小写敏感的

3. **不指定 splits**：如果不指定 `--splits_file` 和 `--splits`，会使用所有可用数据

4. **数据量差异**：
   - Train: 753 samples（较大，慢）
   - Val: 94 samples（中等，快）
   - Test: 95 samples（中等，快）

### 最佳实践

**推荐工作流程：**

```bash
# 1. 开发阶段：在验证集上快速迭代
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits val \
    --output_dir results/dev_val

# 2. 最终评估：在测试集上评估
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits test \
    --output_dir results/final_test

# 3. 过拟合检测：比较训练集和测试集
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits train \
    --output_dir results/check_train

bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits test \
    --output_dir results/check_test

# 比较 Top-1 准确率，如果训练集明显高于测试集，说明可能过拟合
```

---

## 查看结果

### 查看摘要

```bash
# 查看完整摘要
python scripts/view_analysis_summary.py --results_dir results/embedding_analysis

# 只查看特定部分
python scripts/view_analysis_summary.py --sections retrieval distances
```

### 查看可视化

```bash
# Mac
open results/embedding_analysis/visualizations/*.png
open results/embedding_analysis/advanced_analysis/*.png

# Linux
xdg-open results/embedding_analysis/visualizations/*.png
```

### 查看报告

```bash
# 查看 Markdown 报告
cat results/embedding_analysis/visualizations/analysis_report.md

# 或用编辑器打开
code results/embedding_analysis/visualizations/analysis_report.md
```

### 查看数据

```bash
# 使用 pandas 快速查看 CSV
python -c "
import pandas as pd

# 查看距离数据
df = pd.read_csv('results/embedding_analysis/visualizations/pocket_ligand_distances.csv')
print(df.head(10))
print('\nSummary:')
print(df.describe())
"

# 或用 Excel/Numbers 打开
open results/embedding_analysis/visualizations/*.csv
```

---

## 典型工作流程

### 场景 1: 评估新训练的模型

```bash
# 1. 运行分析
bash scripts/run_embedding_analysis.sh \
    --checkpoint models/checkpoints/epoch_200.pt \
    --output_dir results/viz_epoch200

# 2. 查看检索性能
python scripts/view_analysis_summary.py \
    --results_dir results/viz_epoch200 \
    --sections retrieval

# 3. 检查关键指标
# - Top-1 accuracy > 60%?
# - Top-10 accuracy > 80%?
# - MRR > 0.7?
```

### 场景 2: 在测试集上评估模型

```bash
# 只在测试集上运行评估（最常用）
bash scripts/run_embedding_analysis.sh \
    --checkpoint models/checkpoints/best_model.pt \
    --splits_file data/splits/splits.json \
    --splits test \
    --output_dir results/test_set_analysis

# 查看测试集性能
python scripts/view_analysis_summary.py \
    --results_dir results/test_set_analysis \
    --sections retrieval distances
```

### 场景 3: 比较训练集和测试集性能

```bash
# 训练集评估
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits train \
    --output_dir results/train_analysis

# 测试集评估
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits test \
    --output_dir results/test_analysis

# 比较性能
echo "=== Train Set ==="
python scripts/view_analysis_summary.py \
    --results_dir results/train_analysis \
    --sections retrieval | grep "Top-"

echo ""
echo "=== Test Set ==="
python scripts/view_analysis_summary.py \
    --results_dir results/test_analysis \
    --sections retrieval | grep "Top-"
```

### 场景 4: 比较不同训练阶段的模型

```bash
# 为每个模型在测试集上生成结果
for epoch in 50 100 150 200; do
    bash scripts/run_embedding_analysis.sh \
        --checkpoint models/checkpoints/epoch_${epoch}.pt \
        --splits_file data/splits/splits.json \
        --splits test \
        --output_dir results/viz_epoch${epoch}_test
done

# 比较检索性能
for epoch in 50 100 150 200; do
    echo "=== Epoch $epoch (Test Set) ==="
    python scripts/view_analysis_summary.py \
        --results_dir results/viz_epoch${epoch}_test \
        --sections retrieval | grep "Top-"
done
```

### 场景 5: 分析特定配体

```bash
# 1. 在测试集上运行完整分析
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits test

# 2. 筛选特定配体的数据
python -c "
import pandas as pd

df = pd.read_csv('results/embedding_analysis/visualizations/pocket_ligand_distances.csv')

# 只看 ATP 配体
atp_data = df[df['ligand_name'] == 'ATP']
print(f'ATP pockets: {len(atp_data)}')
print(f'Mean cosine similarity: {atp_data[\"cosine_similarity\"].mean():.4f}')
print(atp_data[['pocket_id', 'cosine_similarity']].sort_values('cosine_similarity', ascending=False).head(10))
"
```

---

## 关键输出文件说明

### 必看文件

| 文件 | 用途 | 关键指标 |
|-----|------|---------|
| `analysis_report.md` | 综合报告 | 所有关键指标汇总 |
| `retrieval_results.csv` | 检索性能 | Top-K 准确率 |
| `pocket_ligand_distances.csv` | 距离矩阵 | 余弦相似度 |
| `joint_pca_by_type.png` | PCA 可视化 | 整体分布 |
| `ligand_summary.csv` | 配体统计 | 数据分布 |

### 可视化图片说明

| 图片 | 含义 | 怎么看 |
|-----|------|-------|
| `joint_*_by_type.png` | Pocket vs Ligand 分布 | 两类应该有重叠但可区分 |
| `joint_*_by_ligand.png` | 不同配体的分布 | 同一配体的点应该聚集 |
| `joint_*_connections.png` | Pocket-Ligand 配对 | 连线越短越好 |
| `distance_distributions.png` | 距离分布 | 余弦距离应该较小（<0.3） |
| `retrieval_performance.png` | 检索性能曲线 | 曲线越陡越好 |
| `intra_inter_distances.png` | 配体内/间距离 | 两个分布应该分离 |
| `ligand_similarity_heatmap.png` | 配体相似度 | 发现相似的配体 |

---

## 常见问题

### Q1: 脚本运行很慢怎么办？

**A**: 降维算法（特别是 t-SNE）可能较慢。可以：

```bash
# 只使用 PCA（最快）
bash scripts/run_embedding_analysis.sh --methods pca

# 或减少数据量（随机采样 graph 文件）
```

### Q2: UMAP 报错怎么办？

**A**: 安装 UMAP 或跳过：

```bash
# 安装
pip install umap-learn

# 或跳过 UMAP
bash scripts/run_embedding_analysis.sh --methods "pca tsne"
```

### Q3: 没有匹配的 pocket-ligand 对怎么办？

**A**: 检查文件命名和配体名称：

```python
# 检查 graph 文件命名
import os
print(os.listdir('data/processed/graphs/')[:5])
# 应该看到: ['1aju_ARG_model0.pt', ...]

# 检查 ligand embeddings 键名
import h5py
with h5py.File('data/processed/ligand_embeddings_dedup.h5', 'r') as f:
    print(list(f.keys())[:10])
# 应该看到: ['ARG', 'GTP', 'ATP', ...]
```

### Q4: 如何解读结果好坏？

**A**: 参考以下标准：

| 指标 | 优秀 | 良好 | 一般 | 较差 |
|------|-----|------|------|------|
| Top-1 准确率 | >70% | 50-70% | 30-50% | <30% |
| Top-10 准确率 | >90% | 80-90% | 60-80% | <60% |
| MRR | >0.8 | 0.6-0.8 | 0.4-0.6 | <0.4 |
| 平均余弦距离 | <0.2 | 0.2-0.3 | 0.3-0.4 | >0.4 |
| 余弦相似度 | >0.8 | 0.7-0.8 | 0.6-0.7 | <0.6 |

---

## 自定义分析

### 修改可视化参数

编辑 `visualize_embeddings.py`，找到 `perform_dimensionality_reduction` 函数：

```python
# 修改 t-SNE 参数
if method == 'tsne':
    default_params = {
        'n_components': n_components,
        'perplexity': 50,        # 增大邻域 (默认 30)
        'random_state': 42,
        'n_iter': 2000           # 增加迭代 (默认 1000)
    }

# 修改 UMAP 参数
elif method == 'umap':
    default_params = {
        'n_components': n_components,
        'random_state': 42,
        'n_neighbors': 20,       # 增大邻居数 (默认 15)
        'min_dist': 0.05         # 减小最小距离 (默认 0.1)
    }
```

### 只分析 Top-N 配体

编辑 `visualize_embeddings.py`，在 `match_pocket_ligand_pairs` 中添加：

```python
# 获取 Top-N 配体
from collections import Counter
ligand_counts = Counter([p['ligand_name'] for p in pocket_results.values()])
top_n_ligands = set([lig for lig, _ in ligand_counts.most_common(20)])

# 只保留 Top-N
for pocket_id, pocket_data in pocket_results.items():
    ligand_name = pocket_data['ligand_name']

    if ligand_name not in top_n_ligands:
        continue  # 跳过

    # ... 其余代码
```

---

## 进阶使用

### 导出用于发表的高质量图片

```python
# 在脚本中修改 savefig 参数
plt.savefig(
    output_path,
    dpi=600,              # 提高分辨率（默认 300）
    format='pdf',         # 矢量格式
    bbox_inches='tight',
    transparent=True      # 透明背景
)
```

### 批量处理多个数据集

```bash
#!/bin/bash
for dataset in dataset1 dataset2 dataset3; do
    bash scripts/run_embedding_analysis.sh \
        --graph_dir data/processed/${dataset}/graphs \
        --output_dir results/${dataset}_analysis
done
```

### 生成 HTML 交互式可视化

使用 plotly（需要安装：`pip install plotly`）：

```python
import plotly.express as px

# 在 visualize_embeddings.py 中添加
fig = px.scatter(
    df,
    x='PCA_1',
    y='PCA_2',
    color='Ligand',
    hover_data=['pocket_id', 'ligand_name'],
    title='Interactive PCA Visualization'
)
fig.write_html(output_dir / 'interactive_pca.html')
```

---

## 总结

### 最小化工作流程

```bash
# 1. 运行分析（5-30 分钟，取决于数据量）
bash scripts/run_embedding_analysis.sh

# 2. 查看摘要（1 分钟）
python scripts/view_analysis_summary.py

# 3. 查看可视化（1 分钟）
open results/embedding_analysis/visualizations/*.png
```

### 完整报告

所有关键信息都在 `analysis_report.md` 中，可以：

```bash
# 直接查看
cat results/embedding_analysis/visualizations/analysis_report.md

# 转换为 PDF（需要 pandoc）
pandoc results/embedding_analysis/visualizations/analysis_report.md \
    -o results/embedding_analysis/report.pdf
```

### 获取帮助

```bash
# 查看脚本帮助
python scripts/visualize_embeddings.py --help
python scripts/advanced_embedding_analysis.py --help
bash scripts/run_embedding_analysis.sh --help

# 查看详细文档
cat scripts/README_embedding_visualization.md
```

---

**祝您分析顺利！** 🎉
