# 数据集分组功能（Dataset Splits Feature）

## 功能概述

新增了数据集分组选择功能，可以根据 `data/splits/splits.json` 选择分析特定的数据集（train、val、test 或它们的组合）。

## 快速开始

### 基本用法

```bash
# 分析测试集
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits test

# 分析验证集和测试集
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits "val test"

# 分析所有数据
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits "train val test"
```

### Python 脚本用法

```bash
# 主可视化脚本
python scripts/visualize_embeddings.py \
    --checkpoint models/checkpoints/best_model.pt \
    --graph_dir data/processed/graphs \
    --ligand_embeddings data/processed/ligand_embeddings_dedup.h5 \
    --output_dir results/test_analysis \
    --splits_file data/splits/splits.json \
    --splits test

# 查看帮助
python scripts/visualize_embeddings.py --help
```

## 数据集统计

根据 `data/splits/splits.json`：

| Split | 样本数 | 占比 |
|-------|--------|------|
| Train | 753    | 79.9% |
| Val   | 94     | 10.0% |
| Test  | 95     | 10.1% |
| Total | 942    | 100%  |

## 使用场景

### 1. 模型最终评估（推荐）

```bash
# 只在测试集上评估，避免训练集泄漏
bash scripts/run_embedding_analysis.sh \
    --checkpoint models/checkpoints/best_model.pt \
    --splits_file data/splits/splits.json \
    --splits test \
    --output_dir results/final_evaluation

# 查看结果
python scripts/view_analysis_summary.py \
    --results_dir results/final_evaluation
```

**为什么推荐？**
- 避免在训练数据上评估（防止过拟合的假象）
- 符合机器学习最佳实践
- 得到真实的泛化性能指标

### 2. 快速迭代开发

```bash
# 在较小的验证集上快速测试
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits val \
    --output_dir results/dev_iteration
```

**优势：**
- 验证集只有 94 个样本，运行速度快
- 可以快速迭代调试
- 节省计算资源

### 3. 过拟合检测

```bash
# 比较训练集和测试集性能
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits train \
    --output_dir results/train_perf

bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits test \
    --output_dir results/test_perf

# 比较关键指标
echo "=== Train Set ==="
python scripts/view_analysis_summary.py \
    --results_dir results/train_perf \
    --sections retrieval | grep "Top-1"

echo "=== Test Set ==="
python scripts/view_analysis_summary.py \
    --results_dir results/test_perf \
    --sections retrieval | grep "Top-1"
```

**判断标准：**
- 如果训练集 Top-1 = 90%，测试集 Top-1 = 60% → **严重过拟合**
- 如果训练集 Top-1 = 75%，测试集 Top-1 = 70% → **轻微过拟合，可接受**
- 如果训练集 Top-1 = 70%，测试集 Top-1 = 72% → **良好泛化**

### 4. 模型版本比较

```bash
# 在测试集上比较不同训练阶段的模型
for epoch in 50 100 150 200; do
    bash scripts/run_embedding_analysis.sh \
        --checkpoint models/checkpoints/epoch_${epoch}.pt \
        --splits_file data/splits/splits.json \
        --splits test \
        --output_dir results/epoch_${epoch}_test
done

# 生成性能对比表
echo "Epoch,Top1,Top5,Top10,MRR" > results/model_comparison.csv
for epoch in 50 100 150 200; do
    echo -n "$epoch," >> results/model_comparison.csv
    python scripts/view_analysis_summary.py \
        --results_dir results/epoch_${epoch}_test \
        --sections retrieval | grep "Top-" | awk '{print $2}' | tr '\n' ',' >> results/model_comparison.csv
    echo "" >> results/model_comparison.csv
done
```

### 5. 交叉验证分析

```bash
# 分别在 val 和 test 上评估，验证稳定性
bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits val \
    --output_dir results/val_eval

bash scripts/run_embedding_analysis.sh \
    --splits_file data/splits/splits.json \
    --splits test \
    --output_dir results/test_eval

# 如果 val 和 test 性能接近，说明模型稳定
```

## 技术细节

### 工作原理

1. **加载 splits.json**：读取指定分组的 complex ID 列表
2. **过滤 graph 文件**：只保留在指定分组中的 graph 文件
3. **推理和分析**：对过滤后的数据进行完整的推理和分析流程

### 文件匹配

脚本通过文件名（去除 .pt 扩展名）与 splits.json 中的 ID 进行匹配：

```python
# splits.json
{
  "test": [
    "1aju_ARG_model0",  # ← 这个 ID
    "2kx8_GTP_model1",
    ...
  ]
}

# Graph 文件
data/processed/graphs/1aju_ARG_model0.pt  # ← 匹配这个文件
```

### 代码实现

核心函数位于 `scripts/visualize_embeddings.py`:

```python
def load_splits(splits_file, split_names):
    """加载数据集分组"""
    with open(splits_file, 'r') as f:
        splits_data = json.load(f)

    selected_ids = set()
    for split_name in split_names:
        if split_name in splits_data:
            selected_ids.update(splits_data[split_name])

    return selected_ids

def batch_inference_with_metadata(..., selected_ids=None):
    """批量推理，支持 split 过滤"""
    graph_files = Path(graph_dir).glob("*.pt")

    if selected_ids is not None:
        graph_files = [f for f in graph_files if f.stem in selected_ids]

    # 继续推理...
```

## 参数说明

### --splits_file

- **类型**: 字符串（文件路径）
- **必需**: 否（默认不使用 splits）
- **说明**: splits.json 文件的路径
- **示例**: `data/splits/splits.json`

### --splits

- **类型**: 字符串列表
- **必需**: 否（默认不使用 splits）
- **可选值**: `train`, `val`, `test`（可多选）
- **说明**: 要分析的数据集分组
- **示例**:
  - 单个: `--splits test`
  - 多个: `--splits "val test"`
  - 全部: `--splits "train val test"`

**注意**：必须同时指定 `--splits_file` 和 `--splits` 才会启用过滤。

## 验证和测试

### 测试脚本

```bash
# 运行测试脚本，验证功能正常
python scripts/test_splits_functionality.py
```

测试内容：
- ✓ 加载 splits.json
- ✓ 验证文件结构
- ✓ 测试不同组合
- ✓ 检查与 graph 文件的匹配情况

### 预期输出

```
Split Statistics:
  train   :  753 samples
  val     :   94 samples
  test    :   95 samples
  Total   :  942 samples

Testing Split Combinations:
  Train only          :  753 samples
  Val only            :   94 samples
  Test only           :   95 samples
  Val + Test          :  189 samples
  Train + Val         :  847 samples
  All splits          :  942 samples

✓ All tests passed!
```

## 常见问题

### Q1: 为什么匹配不到任何文件？

**A**: 检查以下几点：
1. splits.json 的路径是否正确
2. graph 文件目录是否正确
3. ID 格式是否匹配（大小写敏感）
4. graph 文件名格式是否正确（应该是 `{id}.pt`）

### Q2: 可以自定义 splits.json 吗？

**A**: 可以！格式如下：

```json
{
  "my_custom_split": [
    "complex_id_1",
    "complex_id_2",
    ...
  ]
}
```

然后使用：
```bash
--splits_file your_splits.json --splits my_custom_split
```

### Q3: 不使用 splits 功能会怎样？

**A**: 如果不指定 `--splits_file` 和 `--splits`，脚本会处理 graph_dir 中的所有 .pt 文件，与之前的行为完全一致。

### Q4: 可以混合使用多个 split 吗？

**A**: 可以！使用空格分隔：

```bash
--splits "train val test"  # 分析全部数据
--splits "val test"        # 分析 val + test
```

### Q5: 如何确认过滤是否生效？

**A**: 查看输出日志：

```
Found 942 pocket graphs in data/processed/graphs
Filtered to 95 graphs based on split selection (removed 847)
```

## 性能考虑

### 数据量对比

| 配置 | 样本数 | 预计时间* |
|------|--------|----------|
| 全部数据 | 942 | ~15-30 分钟 |
| 训练集 | 753 | ~12-25 分钟 |
| 验证集 | 94 | ~2-5 分钟 |
| 测试集 | 95 | ~2-5 分钟 |
| Val+Test | 189 | ~4-8 分钟 |

*时间估计取决于硬件配置和是否使用 GPU

### 优化建议

1. **开发阶段**: 使用 `--splits val` 快速迭代
2. **最终评估**: 使用 `--splits test` 获取真实性能
3. **完整分析**: 使用 `--splits "train val test"` 或不指定 splits

## 相关文件

- `scripts/visualize_embeddings.py` - 主可视化脚本（包含 splits 功能）
- `scripts/run_embedding_analysis.sh` - 一键运行脚本
- `scripts/test_splits_functionality.py` - 测试脚本
- `scripts/QUICKSTART_visualization.md` - 快速入门指南
- `data/splits/splits.json` - 数据集分组文件

## 更新日志

### v1.1.0 (当前版本)
- ✨ 新增数据集分组选择功能
- ✨ 支持 train/val/test 任意组合
- ✨ 添加 `--splits_file` 和 `--splits` 参数
- 📝 更新文档和示例
- ✅ 添加测试脚本

### v1.0.0 (原始版本)
- ✨ 基础可视化和分析功能
- ✨ PCA/t-SNE/UMAP 降维
- ✨ 检索性能评估
- ✨ 聚类分析

## 贡献

如有问题或建议，请查阅：
- 详细文档: `scripts/README_embedding_visualization.md`
- 快速入门: `scripts/QUICKSTART_visualization.md`
- 主脚本: `scripts/visualize_embeddings.py`

---

**Happy Analyzing! 🎉**
