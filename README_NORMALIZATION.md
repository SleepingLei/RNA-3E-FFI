# 数据标准化 - 快速参考

## 🎯 快速开始

### 训练数据准备（自动标准化）

```bash
# 1. 生成配体embeddings（自动标准化）
python scripts/02_embed_ligands.py \
    --complexes_csv hariboss/Complexes.csv \
    --compounds_csv hariboss/compounds.csv \
    --output_h5 data/processed/ligand_embeddings.h5

# 2. 构建图数据集（自动标准化）
python scripts/03_build_dataset.py \
    --hariboss_csv hariboss/Complexes.csv \
    --amber_dir data/processed/amber \
    --output_dir data/processed/graphs
```

### 对已有数据进行标准化

```bash
# 标准化embeddings
python scripts/normalize_embeddings.py \
    --input data/processed/ligand_embeddings.h5 \
    --inplace  # 或者不加 --inplace 创建新文件

# 标准化图特征
python scripts/normalize_graphs.py \
    --graph-dir data/processed/graphs \
    --inplace  # 或者不加 --inplace 创建新目录
```

### 推理时使用

```python
from normalization_utils import NormalizationContext

with NormalizationContext('data/processed') as norm:
    # 使用训练时的标准化参数
    normalized_features = norm.normalize_features(test_features)
    normalized_embedding = norm.normalize_embedding(test_embedding)

    # 模型推理
    predictions = model(normalized_features, normalized_embedding)
```

## 📁 生成的文件

```
data/processed/
├── ligand_embeddings.h5                    # 标准化后的配体embeddings
├── ligand_embedding_norm_params.npz        # 配体标准化参数 ⭐
├── graphs/                                 # 标准化后的图数据
│   └── *.pt
└── node_feature_norm_params.npz            # 节点特征标准化参数 ⭐
```

⭐ = **必须保存！推理时需要使用**

## 🛠️ 工具脚本

| 脚本 | 功能 | 用途 |
|------|------|------|
| `scripts/02_embed_ligands.py` | 生成+标准化embeddings | 训练数据准备 |
| `scripts/03_build_dataset.py` | 构建+标准化图 | 训练数据准备 |
| `scripts/normalize_embeddings.py` | 标准化已有embeddings | 后处理已有数据 |
| `scripts/normalize_graphs.py` | 标准化已有图 | 后处理已有数据 |
| `normalization_utils.py` | 标准化工具函数 | 推理/测试阶段 |

## 🔍 检查数据

```bash
# 检查embeddings
python scripts/normalize_embeddings.py \
    --input data/processed/ligand_embeddings.h5 \
    --inspect-only

# 检查图数据
python scripts/normalize_graphs.py \
    --graph-dir data/processed/graphs \
    --inspect-only
```

**标准化后的输出应该显示:**
- Mean: ~0.0
- Std: ~1.0
- Appears normalized: ✓ Yes

## ⚠️ 重要提醒

1. **训练时**: 计算并保存标准化参数
2. **测试时**: 使用训练时的标准化参数（不要重新计算！）
3. **推理时**: 使用训练时的标准化参数
4. **备份**: 使用 `--inplace` 前先备份原始数据

## 📚 详细文档

详细使用指南请参考: [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md)

## 🧪 测试

```bash
# 运行测试验证功能
python test_normalization.py
```

## 💡 常见用法

### 场景1: 首次训练
```bash
# 直接运行数据准备脚本，标准化会自动完成
python scripts/02_embed_ligands.py ...
python scripts/03_build_dataset.py ...
```

### 场景2: 已有未标准化的数据
```bash
# 使用独立的标准化脚本
python scripts/normalize_embeddings.py --input embeddings.h5 --inplace
python scripts/normalize_graphs.py --graph-dir graphs/ --inplace
```

### 场景3: 新数据推理
```python
from normalization_utils import NormalizationContext

with NormalizationContext('data/processed') as norm:
    # 应用训练时的标准化参数到新数据
    norm_features = norm.normalize_features(new_features)
    norm_embedding = norm.normalize_embedding(new_embedding)
    prediction = model(norm_features, norm_embedding)
```

## 📊 示例代码

完整的推理示例: [example_inference_with_normalization.py](example_inference_with_normalization.py)

```bash
# 运行示例（需要先有标准化参数文件）
python example_inference_with_normalization.py
```

---

**问题?** 查看 [NORMALIZATION_GUIDE.md](NORMALIZATION_GUIDE.md) 获取详细信息
