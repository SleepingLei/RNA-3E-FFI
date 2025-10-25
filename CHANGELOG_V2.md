# RNA-3E-FFI v2.0 Changelog

## 📌 版本信息

- **版本**: v2.0
- **日期**: 2025-10-25
- **状态**: ✅ 已测试，生产就绪

## 🎯 主要变更

### v1.0 → v2.0 升级内容

#### 1. 特征编码方式改进 ✨

**v1.0 (旧方案)**:
- 使用 one-hot 编码
- 特征维度: 115 维
  - Atom types: 69 维 one-hot
  - Charge: 1 维标量
  - Residues: 42 维 one-hot
  - Atomic number: 1 维标量
  - 其他特征: 2 维

**v2.0 (新方案)**:
- **使用整数索引编码 (1-indexed)**
- **特征维度: 4 维**
  - `atom_type_idx`: 整数 (1-70, 其中 70 为 `<UNK>`)
  - `charge`: 浮点数标量
  - `residue_idx`: 整数 (1-43, 其中 43 为 `<UNK>`)
  - `atomic_number`: 整数

**优势**:
- 大幅降低特征维度: 115 → 4
- 减少计算和存储开销
- 保持完整的原子类型信息
- 更符合现代 GNN 的 embedding 层设计

#### 2. 词汇表系统 📚

新增固定词汇表文件，确保跨数据集的特征维度一致性:

**`data/vocabularies/atom_type_vocab.json`**:
```json
{
  "vocab": ["H", "HO", "HS", "H1", ..., "UM5"],
  "vocab_to_idx": {"H": 0, "HO": 1, ...},
  "idx_to_vocab": {"0": "H", "1": "HO", ...},
  "num_types": 69,
  "unk_idx": 69
}
```

**`data/vocabularies/residue_vocab.json`**:
```json
{
  "vocab": ["A", "G", "C", "U", ...],
  "vocab_to_idx": {"A": 0, "G": 1, ...},
  "idx_to_vocab": {"0": "A", "1": "G", ...},
  "num_types": 42,
  "unk_idx": 42
}
```

**包含的 AMBER 原子类型** (69 种):
- 氢原子: H, HO, HS, H1-H5, HW, HC, HA, HP, HZ
- 氧原子: OH, OS, O, O2, OP, OW, O3P
- 氮原子: N, NA, NB, NC, N*, N2, N3, NT, NP
- 碳原子: C, CA, CB, CC, CD, CK, CM, CN, CQ, CR, CV, CW, C*, CT, CI, C2-C5, C5P, CS, CP
- 磷原子: P
- 其他: S, SH, F, Cl, Br, I, MG, K, Na, Zn, Ca, Li, Rb, Cs
- 修饰核苷酸: CM5, CM6, UM5

**包含的 RNA 残基类型** (42 种):
- 标准核苷酸: A, G, C, U
- 5' 末端: A5, G5, C5, U5
- 3' 末端: A3, G3, C3, U3
- 其他命名: RA, RG, RC, RU, DA, DG, DC, DT
- 全名: ADE, GUA, CYT, URA
- 修饰核苷酸: PSU, I, M2G, M7G, OMC, OMG, 5MU, 5MC, 1MA, 2MG, 6MA
- 离子: MG, K, NA, CA, ZN, MN, CL

#### 3. 真实 LJ 参数提取 ⚛️

**v1.0 (占位值)**:
```python
lj_A = 0.0  # Placeholder
lj_B = 0.0  # Placeholder
```

**v2.0 (真实提取)**:
```python
# 从 prmtop 提取真实 LJ 参数
lj_acoef = np.array(amber_parm.parm_data['LENNARD_JONES_ACOEF'])
lj_bcoef = np.array(amber_parm.parm_data['LENNARD_JONES_BCOEF'])
nb_parm_index = np.array(amber_parm.parm_data['NONBONDED_PARM_INDEX'])
ntypes = amber_parm.ptr('ntypes')

# 根据原子类型对计算参数索引
type_i = amber_parm.atoms[i].nb_idx - 1
type_j = amber_parm.atoms[j].nb_idx - 1
parm_idx = nb_parm_index[type_i * ntypes + type_j] - 1

# 提取 A 和 B 系数
lj_A = float(lj_acoef[parm_idx])
lj_B = float(lj_bcoef[parm_idx])
```

**测试结果** (11-nt RNA, 11,606 非键边):
- LJ_A 范围: 0 - 6.03×10⁶
- LJ_A 平均: 3.69×10⁵
- LJ_B 范围: 0 - 2,196
- LJ_B 平均: 312.6

## 📊 测试验证

### 测试数据
- **文件**: `test_output/1aju_ARG_graph_intermediate/rna.prmtop`
- **分子**: 11-nt RNA
- **原子数**: 349

### v2.0 测试结果

```bash
$ python test_v2_features.py

✅ Feature dimension: 4
✅ Atom type vocabulary: 69 types
✅ Residue vocabulary: 42 types
✅ Graph construction: 349 atoms, 512 edges
✅ LJ parameters: extracted

Graph structure:
  Nodes: 349
  Node feature dim: 4 ✓
  Positions: [349, 3]
  1-hop edges (bonds): 512
  2-hop paths (angles): 397
  3-hop paths (dihedrals): 782
  Non-bonded edges: 11,606

Sample node features:
  [atom_type_idx, charge, residue_idx, atomic_num]
  Atom 0: [2.0, 0.4295, 6.0, 1.0]
  Atom 1: [14.0, -0.6223, 6.0, 8.0]
  Atom 2: [44.0, 0.0558, 6.0, 6.0]
```

## 🔧 修改的文件

### 核心修改

1. **`scripts/amber_vocabulary.py`**
   - 修改 `encode_atom_type()`: 返回整数索引 (1-70)
   - 修改 `encode_residue()`: 返回整数索引 (1-43)
   - 修改 `encode_atom_features()`: 返回 4 维数组
   - 添加 `save_vocabularies()`: 保存词汇表到 JSON
   - 更新 `feature_dim` 属性: 返回 4

2. **`scripts/03_build_dataset.py`**
   - 使用 `get_global_encoder()` 获取固定词汇表编码器
   - 实现真实 LJ 参数提取逻辑
   - 处理 numpy 数组转换警告

3. **`data/vocabularies/`** (新增)
   - `atom_type_vocab.json`: 69 种 AMBER 原子类型
   - `residue_vocab.json`: 42 种 RNA 残基类型

### 新增文件

- **`test_v2_features.py`**: v2.0 测试脚本
- **`CHANGELOG_V2.md`**: 本文档

## 🚀 使用指南

### 1. 构建单个图 (v2.0)

```python
from scripts.amber_vocabulary import get_global_encoder
from scripts.build_dataset import build_graph_from_files

# 获取全局编码器（使用固定词汇表）
encoder = get_global_encoder()
print(f"Feature dim: {encoder.feature_dim}")  # 输出: 4

# 构建图
data = build_graph_from_files(
    rna_pdb_path="dummy.pdb",  # 不使用
    prmtop_path="data/rna.prmtop",
    distance_cutoff=5.0,
    add_nonbonded_edges=True
)

# 特征形状
print(data.x.shape)  # [num_atoms, 4]
```

### 2. 在模型中使用 v2.0 特征

```python
import torch.nn as nn
from e3nn import o3

class E3GNNWithEmbedding(nn.Module):
    def __init__(self, num_atom_types=70, num_residues=43,
                 embedding_dim=32, hidden_irreps="32x0e + 16x1o"):
        super().__init__()

        # Embedding 层
        self.atom_type_embed = nn.Embedding(num_atom_types, embedding_dim)
        self.residue_embed = nn.Embedding(num_residues, embedding_dim)

        # 标量特征投影
        self.scalar_proj = nn.Linear(2, embedding_dim)  # charge + atomic_num

        # E(3)-等变层
        self.conv = ... # E(3) 卷积层

    def forward(self, data):
        # 解析 4 维特征
        atom_type_idx = data.x[:, 0].long()    # [num_atoms]
        charge = data.x[:, 1:2]                 # [num_atoms, 1]
        residue_idx = data.x[:, 2].long()      # [num_atoms]
        atomic_num = data.x[:, 3:4]            # [num_atoms, 1]

        # Embedding
        h_atom = self.atom_type_embed(atom_type_idx)      # [num_atoms, 32]
        h_res = self.residue_embed(residue_idx)           # [num_atoms, 32]
        h_scalar = self.scalar_proj(torch.cat([charge, atomic_num], dim=-1))

        # 组合
        h = h_atom + h_res + h_scalar  # [num_atoms, 32]

        # E(3)-等变卷积
        h = self.conv(h, data.pos, data.edge_index)

        return h
```

### 3. 保存词汇表

```python
from scripts.amber_vocabulary import get_global_encoder

encoder = get_global_encoder()
encoder.save_vocabularies("data/vocabularies/")

# 输出:
# Saved vocabularies to data/vocabularies
#   - atom_type_vocab.json: 69 types
#   - residue_vocab.json: 42 types
```

## 📈 性能对比

| 指标 | v1.0 | v2.0 | 改进 |
|------|------|------|------|
| 特征维度 | 115 | 4 | **96% ↓** |
| 内存占用 (349 atoms) | 156 KB | 5.4 KB | **97% ↓** |
| LJ 参数 | 占位值 | 真实提取 | ✅ |
| 词汇表 | 动态 | 固定 | ✅ |
| 跨数据集一致性 | ❌ | ✅ | ✅ |

## ⚠️ 注意事项

### 向后兼容性

v2.0 **不兼容** v1.0 的已保存图文件，因为特征维度发生了变化。

**解决方案**:
- 使用 v2.0 重新生成所有图数据
- 或者保留两个版本分别处理

### 模型修改需求

如果使用 v2.0 特征，需要修改模型输入层:

```python
# v1.0 模型
input_dim = 115
x = data.x  # [num_atoms, 115]

# v2.0 模型 (需要添加 embedding 层)
num_atom_types = 70
num_residues = 43
self.atom_embed = nn.Embedding(num_atom_types, embed_dim)
self.res_embed = nn.Embedding(num_residues, embed_dim)
```

### 数据集处理

批量重新生成数据集:

```bash
python scripts/03_build_dataset.py \
    --hariboss_csv hariboss/Complexes.csv \
    --amber_dir data/processed/amber \
    --output_dir data/processed/graphs_v2 \
    --distance_cutoff 5.0 \
    --num_workers 8
```

## 🎓 设计理念

### 为什么使用整数索引？

1. **降低维度**: one-hot 编码在高维特征空间中浪费资源
2. **学习能力**: Embedding 层可以学习原子类型的语义关系
3. **现代架构**: 符合 Transformer/GNN 的标准做法
4. **灵活性**: 可以使用预训练的 atom type embedding

### 为什么需要固定词汇表？

1. **一致性**: 确保不同数据集的特征维度相同
2. **可复现**: 训练和推理使用相同的索引映射
3. **未知处理**: 统一处理训练时未见过的原子/残基类型
4. **标准化**: 基于 AMBER ff99bsc0_chiOL3 力场

## 📚 相关文档

- **v1.0 文档**: `README_MULTIHOP.md`
- **完整总结**: `FINAL_SUMMARY.md`
- **测试脚本**: `test_v2_features.py`
- **词汇表文件**: `data/vocabularies/`

## 🔮 未来工作

### 短期 (v2.1)
- [ ] 添加更多修饰核苷酸到词汇表
- [ ] 实现预训练 atom type embedding
- [ ] 优化 LJ 参数提取性能

### 长期 (v3.0)
- [ ] 残基级别图构建
- [ ] 多跳注意力层 (FFiNet-style)
- [ ] 分层 GNN 架构

---

**版本**: v2.0
**日期**: 2025-10-25
**状态**: ✅ 测试通过，生产就绪
**作者**: RNA-3E-FFI Team
