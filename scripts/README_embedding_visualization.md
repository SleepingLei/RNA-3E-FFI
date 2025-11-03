# Embedding Visualization and Analysis Scripts

这套脚本用于分析和可视化 RNA pocket embeddings 和 ligand embeddings 的关系。

## 脚本概览

### 1. `visualize_embeddings.py` - 主可视化脚本

**功能**:
- 对所有 pocket graphs 进行批量推理，生成 pocket embeddings
- 加载对应的 ligand embeddings
- 匹配 pocket-ligand 配对
- 执行多种降维方法（PCA, t-SNE, UMAP）
- 创建联合可视化
- 分析距离和相关性
- 生成综合报告

**输出**:
- 9 种可视化图（3种降维方法 × 3种视角）
- 6 个数据文件（CSV/JSON/NPZ）
- 1 份分析报告（Markdown）

### 2. `advanced_embedding_analysis.py` - 高级分析脚本

**功能**:
- 聚类分析（K-Means，寻找最优聚类数）
- 检索性能评估（Top-K准确率，MRR）
- 配体内/配体间距离分析
- 配体相似度热图
- 层次聚类树状图

**输出**:
- 聚类分析结果
- 检索性能指标
- 距离分布分析
- 配体关系可视化

---

## 使用流程

### 前置条件

1. **训练好的模型检查点**
   ```
   models/checkpoints/best_model.pt
   ```

2. **Pocket graph 文件**（来自 `generate_pocket_graph.py`）
   ```
   data/processed/graphs/
   ├── 1aju_ARG_model0.pt
   ├── 1aju_ARG_model1.pt
   ├── 2kx8_GTP_model0.pt
   └── ...
   ```

3. **去重的 ligand embeddings**（来自 `deduplicate_ligand_embeddings.py`）
   ```
   data/processed/ligand_embeddings_dedup.h5
   ```
   键名格式：`ARG`, `GTP`, `ATP` 等

### 步骤 1: 主可视化分析

```bash
python scripts/visualize_embeddings.py \
    --checkpoint models/checkpoints/best_model.pt \
    --graph_dir data/processed/graphs \
    --ligand_embeddings data/processed/ligand_embeddings_dedup.h5 \
    --output_dir results/visualizations \
    --methods pca tsne umap
```

**参数说明**:
- `--checkpoint`: 训练好的模型检查点路径
- `--graph_dir`: 包含 pocket graph 文件的目录
- `--ligand_embeddings`: 去重的 ligand embeddings HDF5 文件
- `--output_dir`: 输出目录（默认：`results/visualizations`）
- `--methods`: 降维方法列表（可选：`pca`, `tsne`, `umap`）
- `--device`: 设备选择（`cuda`/`cpu`，默认自动检测）

**输出文件**:

```
results/visualizations/
├── Data Files
│   ├── pocket_embeddings.npz              # 所有 pocket embeddings
│   ├── matched_pairs.json                 # 匹配的 pocket-ligand 对
│   ├── pocket_ligand_distances.csv        # 每对的距离指标
│   ├── pocket_ligand_correlations.csv     # 每对的相关性指标
│   ├── ligand_summary.csv                 # 配体统计汇总
│   └── ligand_distance_stats.csv          # 按配体类型的距离统计
│
├── Visualizations (3 methods × 3 views)
│   ├── joint_pca_by_type.png/pdf          # PCA: 按类型着色（pocket/ligand）
│   ├── joint_pca_by_ligand.png/pdf        # PCA: 按配体名称着色
│   ├── joint_pca_connections.png/pdf      # PCA: 显示 pocket-ligand 连接
│   ├── joint_tsne_*.png/pdf               # t-SNE 对应的 3 种视图
│   ├── joint_umap_*.png/pdf               # UMAP 对应的 3 种视图
│   ├── distance_distributions.png/pdf     # 距离分布图
│   ├── correlation_distributions.png/pdf  # 相关性分布图
│   └── ligand_distribution.png/pdf        # 配体频率分布
│
└── analysis_report.md                     # 综合分析报告
```

### 步骤 2: 高级分析（可选）

在完成步骤 1 后，使用生成的 `matched_pairs.json` 进行更深入的分析：

```bash
python scripts/advanced_embedding_analysis.py \
    --matched_pairs results/visualizations/matched_pairs.json \
    --output_dir results/advanced_analysis
```

**输出文件**:

```
results/advanced_analysis/
├── clustering_optimization.png            # 聚类数优化曲线（肘部法则）
├── kmeans_clusters.png                    # K-Means 聚类可视化
├── cluster_assignments.csv                # 聚类分配结果
├── retrieval_performance.png              # 检索性能曲线
├── retrieval_results.csv                  # 详细检索结果
├── intra_inter_distances.png              # 配体内/配体间距离分布
├── intra_inter_boxplot.png                # 距离箱线图比较
├── intra_inter_distances.csv              # 距离数据
├── ligand_similarity_heatmap.png          # Top 20 配体相似度热图
├── ligand_distance_matrix.csv             # 配体距离矩阵
└── ligand_dendrogram.png                  # Top 30 配体层次聚类树
```

---

## 输出解读

### 1. 联合可视化（Joint Visualizations）

#### 按类型着色（by_type）
- **蓝色圆点**: Pocket embeddings
- **红色三角**: Ligand embeddings
- **目的**: 观察 pocket 和 ligand 在嵌入空间中的整体分布

#### 按配体着色（by_ligand）
- **不同颜色**: 不同配体类型
- **圆点**: 该配体对应的 pockets
- **星号**: 该配体的 embedding
- **目的**: 观察特定配体的 pockets 是否聚集在一起

#### 连接视图（connections）
- **灰色线**: 连接每个 pocket 和其对应的 ligand
- **目的**: 直观显示 pocket-ligand 配对的嵌入空间距离

### 2. 距离分析

#### Cosine Distance（余弦距离）
- **范围**: 0 (完全相同) 到 2 (完全相反)
- **解读**: 值越小，pocket 和 ligand embedding 越相似
- **期望**: 如果模型训练良好，同一对的距离应该较小

#### Euclidean Distance（欧氏距离）
- **范围**: 0 到 ∞
- **解读**: 欧氏空间中的直线距离
- **用途**: 补充余弦距离，提供绝对距离信息

#### Cosine Similarity（余弦相似度）
- **范围**: -1 到 1
- **解读**: 1 = 完全相同，0 = 正交，-1 = 完全相反
- **期望**: 对于匹配的 pocket-ligand 对，应该接近 1

### 3. 检索性能指标

#### Top-K Accuracy
- **Top-1**: 真实配体排第一的比例
- **Top-5**: 真实配体排前五的比例
- **Top-10**: 真实配体排前十的比例
- **解读**: 值越高越好，表示模型检索能力强

#### Mean Reciprocal Rank (MRR)
- **公式**: `MRR = mean(1 / rank)`
- **范围**: 0 到 1
- **解读**: 值越高越好，1 表示所有真实配体都排第一

#### 示例解读
```
Top-1 Accuracy:  0.650 (65.0%)  ← 65% 的 pockets 正确检索到真实配体
Top-5 Accuracy:  0.820 (82.0%)  ← 82% 的真实配体在前 5 名内
Mean Rank:       3.24            ← 平均排名第 3.24 位
MRR:             0.712           ← 倒数排名的平均值
```

### 4. 配体内/配体间距离

#### Intra-ligand Distance（配体内距离）
- **定义**: 结合同一配体的不同 pockets 之间的距离
- **期望**: 较小，说明模型学到了配体特异性

#### Inter-ligand Distance（配体间距离）
- **定义**: 结合不同配体的 pockets 之间的距离
- **期望**: 较大，说明模型能区分不同配体

#### 理想情况
```
Intra-ligand mean:  0.15  ← 小
Inter-ligand mean:  0.45  ← 大
```
这说明模型能很好地将相同配体的 pockets 聚在一起，同时区分不同配体。

### 5. 聚类分析

#### Silhouette Score（轮廓系数）
- **范围**: -1 到 1
- **解读**:
  - `> 0.7`: 聚类结构强
  - `0.5-0.7`: 聚类结构合理
  - `0.25-0.5`: 聚类结构弱
  - `< 0.25`: 无明显聚类结构

#### 聚类组成
查看每个聚类中主要包含哪些配体，帮助理解：
- 哪些配体的 pockets 相似？
- 模型是否按配体类型自然聚类？

---

## 高级用法

### 自定义降维参数

```python
# 在脚本中修改 perform_dimensionality_reduction 的 kwargs

# t-SNE
reduced = perform_dimensionality_reduction(
    embeddings, labels, method='tsne',
    perplexity=50,      # 调整邻域大小
    n_iter=2000,        # 增加迭代次数
    learning_rate=200
)

# UMAP
reduced = perform_dimensionality_reduction(
    embeddings, labels, method='umap',
    n_neighbors=20,     # 邻居数
    min_dist=0.05,      # 最小距离
    metric='cosine'     # 距离度量
)
```

### 仅分析特定配体

修改 `match_pocket_ligand_pairs` 函数，添加过滤条件：

```python
target_ligands = ['ATP', 'GTP', 'ARG']  # 只分析这些配体

for pocket_id, pocket_data in pocket_results.items():
    ligand_name = pocket_data['ligand_name']

    if ligand_name not in target_ligands:
        continue  # 跳过不感兴趣的配体

    if ligand_name in ligand_embeddings:
        matched_data.append({...})
```

### 批量处理多个模型

```bash
#!/bin/bash
# 比较不同训练阶段的模型

for epoch in 50 100 150 200; do
    python scripts/visualize_embeddings.py \
        --checkpoint models/checkpoints/model_epoch_${epoch}.pt \
        --graph_dir data/processed/graphs \
        --ligand_embeddings data/processed/ligand_embeddings_dedup.h5 \
        --output_dir results/visualizations_epoch${epoch}
done
```

---

## 故障排除

### 问题 1: "No matched pocket-ligand pairs found"

**原因**: Pocket graph 文件名中的配体名称与 ligand embeddings 中的键不匹配

**解决方案**:
1. 检查 pocket graph 文件命名格式：
   ```bash
   ls data/processed/graphs/ | head
   # 应该看到：1aju_ARG_model0.pt
   ```

2. 检查 ligand embeddings 键名：
   ```python
   import h5py
   with h5py.File('data/processed/ligand_embeddings_dedup.h5', 'r') as f:
       print(list(f.keys())[:10])
   # 应该看到：['ARG', 'GTP', 'ATP', ...]
   ```

3. 确认配体名称提取逻辑：
   ```python
   from scripts.deduplicate_ligand_embeddings import extract_ligand_name
   print(extract_ligand_name('1aju_ARG_model0'))  # 应输出: ARG
   ```

### 问题 2: UMAP 不可用

**原因**: 未安装 UMAP

**解决方案**:
```bash
pip install umap-learn
```

或者只使用 PCA 和 t-SNE：
```bash
python scripts/visualize_embeddings.py \
    ... \
    --methods pca tsne
```

### 问题 3: 内存不足

**原因**: Embeddings 太多或降维算法内存占用大

**解决方案**:
1. 减少 graph 文件数量（采样）
2. 只使用 PCA（内存占用最小）
3. 对于 t-SNE，减少 `n_iter` 参数
4. 对于大数据集，考虑分批处理

### 问题 4: 可视化图片中标签重叠

**解决方案**:
修改脚本中的图片大小和字体：

```python
# 在可视化函数中
fig, ax = plt.subplots(figsize=(16, 12))  # 增大图片尺寸

# 调整图例位置
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=3)
```

---

## 依赖包

```bash
# 核心依赖
pip install numpy pandas matplotlib seaborn
pip install torch torch-geometric
pip install scikit-learn scipy h5py tqdm

# 可选依赖
pip install umap-learn  # 用于 UMAP 降维
```

---

## 引用和参考

- **PCA**: Principal Component Analysis
- **t-SNE**: van der Maaten & Hinton (2008)
- **UMAP**: McInnes et al. (2018)
- **K-Means**: Lloyd (1982)
- **Silhouette Score**: Rousseeuw (1987)

---

## 常见分析场景

### 场景 1: 评估模型训练质量

**目标**: 检查模型是否学到了有意义的 pocket-ligand 关系

**步骤**:
1. 运行 `visualize_embeddings.py`
2. 查看 `distance_distributions.png`
3. 查看 `analysis_report.md` 中的平均余弦距离

**期望**:
- 平均余弦距离 < 0.3（pocket 和对应 ligand 较近）
- 余弦相似度 > 0.7（高相似性）

### 场景 2: 发现配体相似性

**目标**: 找出哪些配体的结合口袋相似

**步骤**:
1. 运行 `advanced_embedding_analysis.py`
2. 查看 `ligand_similarity_heatmap.png`
3. 查看 `ligand_dendrogram.png`

**解读**:
- 热图中颜色较深的区域表示配体相似
- 树状图中距离近的配体可能有相似的结合模式

### 场景 3: 虚拟筛选验证

**目标**: 评估模型的检索能力（用于虚拟筛选）

**步骤**:
1. 运行 `advanced_embedding_analysis.py`
2. 查看 `retrieval_results.csv`
3. 检查 Top-K accuracy

**期望**:
- Top-1 accuracy > 60%
- Top-10 accuracy > 80%
- MRR > 0.7

### 场景 4: 比较不同训练策略

**目标**: 比较不同模型或训练阶段的性能

**步骤**:
```bash
# 为每个模型生成可视化
for model in model_A model_B model_C; do
    python scripts/visualize_embeddings.py \
        --checkpoint models/${model}.pt \
        --output_dir results/viz_${model}
done

# 比较报告
cat results/viz_*/analysis_report.md | grep "Mean:"
```

---

## 输出示例

### 分析报告示例

```markdown
# Pocket-Ligand Embedding Analysis Report

## Summary Statistics
- **Total matched pocket-ligand pairs**: 1,234
- **Unique ligands**: 89
- **Embedding dimension**: 512

## Distance Metrics

### Cosine Distance
- Mean: 0.2156
- Median: 0.1983
- Std: 0.1124
- Min: 0.0034
- Max: 0.8912

### Cosine Similarity
- Mean: 0.7844
- Median: 0.8017
- Std: 0.1124

## Correlation Analysis
- **Mean Pearson r**: 0.4521
- **Mean Spearman r**: 0.4398

## Top 10 Most Common Ligands
| ligand_name | n_pockets |
|-------------|-----------|
| ATP         | 156       |
| GTP         | 134       |
| ARG         | 98        |
| SAM         | 87        |
...
```

### 检索结果示例

```csv
pocket_id,true_ligand,rank,top1,top1_distance,true_ligand_distance
1aju_ARG_model0,ARG,1,ARG,0.1234,0.1234
2kx8_GTP_model1,GTP,3,ATP,0.1456,0.1789
...
```

---

## 问题反馈

如果遇到问题或需要新功能，请检查：
1. 数据格式是否正确（文件名、HDF5 键名）
2. 依赖包版本是否兼容
3. 内存和计算资源是否充足
