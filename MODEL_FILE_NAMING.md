# Model File Naming Convention

## 概述

由于 `01_process_data.py` 为每个 PDB 结构的每个 model 生成单独的文件，文件名包含 `_model{N}` 后缀。这个文档说明了整个 pipeline 中的文件命名约定以及如何处理多模型文件。

---

## 文件命名模式

### 1. `scripts/01_process_data.py` 输出

**格式**: `{pdb_id}_{ligand}_model{N}_{type}.{ext}`

示例：
```
1aju_ARG_model0_pocket.pdb
1aju_ARG_model0_rna.pdb
1aju_ARG_model0_rna.prmtop
1aju_ARG_model0_rna.inpcrd
1aju_ARG_model0_rna_cleaned.pdb
```

如果 PDB 只有一个 model：
```
1akx_ARG_model0_rna.pdb
```

### 2. `scripts/02_embed_ligands.py` 输出

**格式**: HDF5 文件中的键为 `{pdb_id}_{ligand}` (无 model 编号)

**原因**: 相同复合物的所有 model 共享同一个配体，因此配体 embedding 不需要区分 model。

示例 HDF5 结构：
```python
ligand_embeddings.h5/
  ├── 1aju_ARG     # shape: (1536,)
  ├── 7ych_GTP     # shape: (1536,)
  └── 1akx_ARG     # shape: (1536,)
```

### 3. `scripts/03_build_dataset.py` 输出

**格式**: `{pdb_id}_{ligand}_model{N}.pt`

每个 model 的 RNA 结构生成独立的图：
```
1aju_ARG_model0.pt
1aju_ARG_model1.pt  # 如果有多个 model
7ych_GTP_model0.pt
```

**兼容性**: 脚本也支持无 model 编号的旧格式：`{pdb_id}_{ligand}.pt`

---

## 各脚本的处理逻辑

### `scripts/03_build_dataset.py`

**查找文件策略**:
1. 首先尝试查找带 model 编号的文件: `{complex_base}_model*_rna.pdb`
2. 如果找不到，回退到无 model 编号: `{complex_base}_rna.pdb`

**代码示例** (第342-360行):
```python
# 查找所有 model 文件
pattern = str(amber_dir / f"{complex_id}_model*_rna.pdb")
model_pdb_files = sorted(glob.glob(pattern))

if not model_pdb_files:
    # 回退: 尝试无 model 编号的文件
    rna_pdb_path = amber_dir / f"{complex_id}_rna.pdb"
    if rna_pdb_path.exists():
        model_pdb_files = [rna_pdb_path]

# 处理每个 model
for rna_pdb_path in model_pdb_files:
    # 提取 model 编号
    if "_model" in stem:
        model_part = stem.split("_model")[1].split("_")[0]
        complex_model_id = f"{complex_id}_model{model_part}"
```

**输出**: 为每个 model 生成独立的 `.pt` 图文件

### `scripts/04_train_model.py`

**处理逻辑**:
1. 从 HARIBOSS CSV 创建 complex base IDs: `{pdb_id}_{ligand}`
2. 查找所有对应的图文件 (可能有多个 model)
3. 建立 graph ID 到 embedding key 的映射

**Graph ID → Embedding Key 映射** (第53-64行):
```python
# complex_id 可能是 "1aju_ARG_model0"
# embedding_key 是 "1aju_ARG"
if '_model' in complex_id:
    base_id = '_'.join(complex_id.split('_model')[0].split('_'))
else:
    base_id = complex_id
```

**数据加载** (第78-93行):
```python
def __getitem__(self, idx):
    complex_id = self.valid_ids[idx]  # e.g., "1aju_ARG_model0"

    # 加载图文件
    graph_path = self.graph_dir / f"{complex_id}.pt"
    data = torch.load(graph_path)

    # 使用映射获取 embedding
    # complex_id 可能是 "1aju_ARG_model0"，但 embedding key 是 "1aju_ARG"
    embedding_key = self.id_to_embedding_key[complex_id]
    ligand_embedding = self.ligand_embeddings[embedding_key]

    data.y = ligand_embedding
    return data
```

**训练效果**:
- 一个复合物有多个 model → 生成多个训练样本
- 所有 model 共享相同的配体 embedding 作为目标
- 提高数据多样性和模型泛化能力

### `scripts/05_run_inference.py`

**单个推理**:
- 用户直接指定图文件路径，支持任何文件名
- 示例: `--query_graph data/processed/graphs/1aju_ARG_model0.pt`

**批量推理**:
- 遍历所有 `.pt` 文件
- 自动处理带或不带 model 编号的文件名
- 参考示例函数 `batch_inference_example()` (第281-326行)

---

## 多模型的优势

### 为什么保留多个 model？

1. **结构多样性**: NMR 结构或同一 PDB 的多个模型提供不同的构象
2. **数据增强**: 相同配体的不同 pocket 结构增加训练数据
3. **泛化能力**: 模型学习到对结构变化的鲁棒性

### 数据流示例

对于 PDB `1aju` 配体 `ARG`，有 2 个 model：

```
输入 (01_process_data.py):
  1aju.cif (包含 model 0 和 model 1)

输出:
  1aju_ARG_model0_rna.pdb/prmtop/inpcrd
  1aju_ARG_model1_rna.pdb/prmtop/inpcrd

配体 embedding (02_embed_ligands.py):
  HDF5['1aju_ARG'] = [1536-dim vector]  # 两个 model 共享

图构建 (03_build_dataset.py):
  1aju_ARG_model0.pt  # Graph 0
  1aju_ARG_model1.pt  # Graph 1

训练 (04_train_model.py):
  训练样本 1: (Graph_1aju_ARG_model0, Embedding_1aju_ARG)
  训练样本 2: (Graph_1aju_ARG_model1, Embedding_1aju_ARG)
  # 相同目标，不同输入 → 学习结构变化的不变性
```

---

## 兼容性

所有脚本都保持向后兼容：

- ✅ 支持新格式: `{pdb_id}_{ligand}_model{N}.pt`
- ✅ 支持旧格式: `{pdb_id}_{ligand}.pt`

如果你的文件是旧格式（无 model 编号），脚本会自动回退并正常工作。

---

## 测试

运行测试脚本验证文件处理逻辑：

```bash
python scripts/test_model_file_handling.py
```

这将：
1. 测试文件名模式匹配逻辑
2. 测试 embedding key 提取
3. 检查实际文件的命名情况
4. 显示完整的文件命名约定总结

---

## 常见问题

### Q: 为什么配体 embedding 不包含 model 编号？
**A**: 配体的化学结构是相同的，只有 RNA pocket 的构象不同。所有 model 应该预测相同的配体 embedding。

### Q: 如果我的数据只有单个 model 怎么办？
**A**: 文件名仍然会有 `_model0` 后缀，这是正常的。脚本会正确处理单模型情况。

### Q: 我可以混合使用有/无 model 编号的文件吗？
**A**: 可以！脚本对两种格式都支持。优先查找带 model 编号的文件，如果找不到则回退到无 model 编号的文件。

### Q: 训练时多个 model 会被当作独立样本吗？
**A**: 是的。每个 model 的图是独立的训练样本，但它们共享相同的目标 embedding。这有助于模型学习结构不变性。

---

## 修改历史

- **2025-10-22**:
  - 修复 `03_build_dataset.py` 支持 `_model{N}` 格式
  - 修复 `04_train_model.py` 支持多模型训练
  - 添加 `05_run_inference.py` 的文档说明
  - 创建测试脚本验证文件处理逻辑
