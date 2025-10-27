# 数据标准化使用指南

## 概述

本项目包含完整的数据标准化流程，用于处理RNA-配体复合物的特征和标签数据。标准化对于防止训练过程中weights变为0以及提高模型性能至关重要。

## 标准化类型

### 1. 节点特征标准化 (Node Feature Normalization)
- **数据**: RNA图的节点特征 (原子特征)
- **方法**: 全局Z-score标准化
- **参数文件**: `data/processed/node_feature_norm_params.npz`
- **包含**: `mean`, `std` (每个特征维度)

### 2. 配体Embedding标准化 (Ligand Embedding Normalization)
- **数据**: 配体的Uni-Mol embeddings (标签)
- **方法**: 全局Z-score标准化
- **参数文件**: `data/processed/ligand_embedding_norm_params.npz`
- **包含**: `mean`, `std` (每个embedding维度)

## 使用场景

### 场景1: 从头开始训练（推荐）

标准化已集成到数据处理pipeline中：

```bash
# 1. 生成配体embeddings (自动标准化)
python scripts/02_embed_ligands.py \
    --complexes_csv hariboss/Complexes.csv \
    --compounds_csv hariboss/compounds.csv \
    --output_h5 data/processed/ligand_embeddings.h5

# 输出:
# - data/processed/ligand_embeddings.h5 (已标准化)
# - data/processed/ligand_embedding_norm_params.npz (标准化参数)

# 2. 构建图数据集 (自动标准化)
python scripts/03_build_dataset.py \
    --hariboss_csv hariboss/Complexes.csv \
    --amber_dir data/processed/amber \
    --output_dir data/processed/graphs

# 输出:
# - data/processed/graphs/*.pt (已标准化的图)
# - data/processed/node_feature_norm_params.npz (标准化参数)
```

✅ **优点**: 自动化，无需额外步骤
⚠️ **注意**: 标准化参数保存在 `data/processed/` 目录

### 场景2: 对已有数据进行标准化

如果你已经生成了未标准化的数据：

#### 2.1 标准化配体Embeddings

```bash
# 检查现有h5文件状态
python scripts/normalize_embeddings.py \
    --input data/processed/ligand_embeddings.h5 \
    --inspect-only

# 创建标准化版本（保留原文件）
python scripts/normalize_embeddings.py \
    --input data/processed/ligand_embeddings.h5 \
    --output data/processed/ligand_embeddings_normalized.h5

# 或直接覆盖原文件（in-place）
python scripts/normalize_embeddings.py \
    --input data/processed/ligand_embeddings.h5 \
    --inplace
```

#### 2.2 标准化图特征

```bash
# 检查现有图文件状态
python scripts/normalize_graphs.py \
    --graph-dir data/processed/graphs \
    --inspect-only

# 创建标准化版本（保留原文件）
python scripts/normalize_graphs.py \
    --graph-dir data/processed/graphs \
    --output-dir data/processed/graphs_normalized

# 或直接覆盖原文件（in-place）
python scripts/normalize_graphs.py \
    --graph-dir data/processed/graphs \
    --inplace
```

### 场景3: 推理/测试阶段

在推理或测试时，**必须**使用训练时保存的标准化参数：

```python
from normalization_utils import NormalizationContext

# 使用上下文管理器自动加载标准化参数
with NormalizationContext('data/processed') as norm:
    # 对新的测试数据应用训练时的标准化参数
    normalized_features = norm.normalize_features(test_node_features)
    normalized_embedding = norm.normalize_embedding(test_ligand_embedding)

    # 使用标准化后的数据进行推理
    predictions = model(normalized_features, normalized_embedding)
```

详细示例请参考: `example_inference_with_normalization.py`

## 工具脚本说明

### 1. `scripts/normalize_embeddings.py`

对HDF5格式的配体embeddings进行标准化。

**参数**:
- `--input`: 输入h5文件路径 (必需)
- `--output`: 输出h5文件路径 (可选，默认添加`_normalized`后缀)
- `--params-output`: 标准化参数保存路径 (可选)
- `--inplace`: 原地修改，覆盖原文件
- `--inspect-only`: 仅检查文件，不进行标准化

**示例**:
```bash
# 基本用法
python scripts/normalize_embeddings.py --input embeddings.h5

# 指定输出位置
python scripts/normalize_embeddings.py \
    --input embeddings.h5 \
    --output embeddings_norm.h5 \
    --params-output custom_params.npz

# 原地修改
python scripts/normalize_embeddings.py --input embeddings.h5 --inplace

# 仅检查
python scripts/normalize_embeddings.py --input embeddings.h5 --inspect-only
```

### 2. `scripts/normalize_graphs.py`

对PyTorch Geometric图数据进行标准化。

**参数**:
- `--graph-dir`: 图文件目录 (必需)
- `--output-dir`: 输出目录 (可选，默认添加`_normalized`后缀)
- `--params-output`: 标准化参数保存路径 (可选)
- `--inplace`: 原地修改，覆盖原文件
- `--inspect-only`: 仅检查文件，不进行标准化
- `--num-samples`: 检查时采样的图数量 (默认10)

**示例**:
```bash
# 基本用法
python scripts/normalize_graphs.py --graph-dir data/processed/graphs

# 指定输出位置
python scripts/normalize_graphs.py \
    --graph-dir data/processed/graphs \
    --output-dir data/processed/graphs_norm \
    --params-output custom_params.npz

# 原地修改
python scripts/normalize_graphs.py --graph-dir data/processed/graphs --inplace

# 仅检查
python scripts/normalize_graphs.py --graph-dir data/processed/graphs --inspect-only
```

### 3. `normalization_utils.py`

Python模块，提供标准化的工具函数。

**主要函数**:
- `load_node_feature_norm_params(path)`: 加载节点特征标准化参数
- `load_ligand_embedding_norm_params(path)`: 加载配体embedding标准化参数
- `normalize_node_features(features, mean, std)`: 标准化节点特征
- `normalize_ligand_embedding(embedding, mean, std)`: 标准化配体embedding
- `denormalize_ligand_embedding(normalized, mean, std)`: 反标准化

**使用示例**:
```python
# 方式1: 手动加载和应用
from normalization_utils import (
    load_node_feature_norm_params,
    normalize_node_features
)

mean, std = load_node_feature_norm_params('data/processed/node_feature_norm_params.npz')
normalized = normalize_node_features(features, mean, std)

# 方式2: 使用上下文管理器
from normalization_utils import NormalizationContext

with NormalizationContext('data/processed') as norm:
    normalized_features = norm.normalize_features(features)
    normalized_embedding = norm.normalize_embedding(embedding)
```

## 文件结构

```
RNA-3E-FFI/
├── scripts/
│   ├── 02_embed_ligands.py           # 生成embeddings (含自动标准化)
│   ├── 03_build_dataset.py           # 构建图数据集 (含自动标准化)
│   ├── normalize_embeddings.py       # 独立的embedding标准化脚本
│   └── normalize_graphs.py           # 独立的图标准化脚本
├── normalization_utils.py            # 标准化工具函数
├── example_inference_with_normalization.py  # 推理示例
└── data/processed/
    ├── ligand_embeddings.h5          # 配体embeddings (标准化后)
    ├── ligand_embedding_norm_params.npz  # 配体标准化参数
    ├── graphs/                       # 图数据 (标准化后)
    │   ├── 1aju_ARG_model0.pt
    │   └── ...
    └── node_feature_norm_params.npz  # 节点特征标准化参数
```

## 验证标准化

### 检查embeddings

```bash
python scripts/normalize_embeddings.py \
    --input data/processed/ligand_embeddings.h5 \
    --inspect-only
```

**预期输出** (标准化后):
- Mean: ~0.0
- Std: ~1.0
- Appears normalized: ✓ Yes

### 检查图特征

```bash
python scripts/normalize_graphs.py \
    --graph-dir data/processed/graphs \
    --inspect-only
```

**预期输出** (标准化后):
- Mean: ~0.0
- Std: ~1.0
- Appears normalized: ✓ Yes

## 常见问题

### Q1: 什么时候需要标准化？
**A**: 始终需要！标准化可以：
- 防止训练过程中weights变为0
- 加速模型收敛
- 提高数值稳定性
- 确保不同特征在相同尺度

### Q2: 训练集和测试集如何处理？
**A**:
- **训练集**: 计算并保存标准化参数，应用到训练数据
- **验证集/测试集**: 使用训练集的标准化参数（不重新计算！）
- **推理**: 使用训练集的标准化参数

### Q3: 已经训练了未标准化的模型怎么办？
**A**:
1. 使用标准化脚本处理所有数据
2. 重新训练模型（推荐）
3. 或者修改推理pipeline保持一致性

### Q4: 标准化参数文件丢失了怎么办？
**A**:
- 如果还有原始数据，重新运行标准化脚本生成参数
- 如果没有原始数据，需要重新处理整个数据集

### Q5: 如何确认标准化是否正确？
**A**: 使用 `--inspect-only` 选项检查：
```bash
python scripts/normalize_embeddings.py --input file.h5 --inspect-only
python scripts/normalize_graphs.py --graph-dir graphs/ --inspect-only
```

### Q6: In-place模式安全吗？
**A**:
- ⚠️ 会覆盖原文件，建议先备份
- 优点：节省磁盘空间
- 建议：首次使用时不要用in-place，确认正确后再使用

## 最佳实践

1. ✅ **总是使用标准化**: 在训练前标准化所有数据
2. ✅ **保存标准化参数**: 确保参数文件与数据一起保存
3. ✅ **一致性**: 训练、验证、测试使用相同的标准化参数
4. ✅ **先检查再处理**: 使用 `--inspect-only` 确认数据状态
5. ✅ **备份原始数据**: 使用in-place前先备份
6. ✅ **文档记录**: 记录使用的标准化参数文件路径

## 技术细节

### Z-score标准化公式

```
x_normalized = (x - mean) / std
```

其中:
- `mean`: 特征/维度的均值（在所有样本上计算）
- `std`: 特征/维度的标准差（在所有样本上计算）
- 对于 `std < 1e-8` 的常量特征，使用 `std = 1.0`

### 为什么是全局标准化？

- **节点特征**: 在所有图的所有原子上计算统计量
- **配体embedding**: 在所有配体上计算统计量
- **原因**: 保证不同样本之间的可比性

### 反标准化

如果需要将标准化后的值还原：

```python
x_original = x_normalized * std + mean
```

示例代码见 `normalization_utils.py` 中的 `denormalize_ligand_embedding()`

## 参考资料

- [PyTorch Geometric文档](https://pytorch-geometric.readthedocs.io/)
- [Feature Scaling Wikipedia](https://en.wikipedia.org/wiki/Feature_scaling)
- 项目内示例: `example_inference_with_normalization.py`

## 联系与支持

如有问题，请查看:
1. 本文档的常见问题部分
2. 代码中的注释和docstrings
3. 示例脚本 `example_inference_with_normalization.py`
