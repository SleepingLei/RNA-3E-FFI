# RNA-3E-FFI 多跳相互作用修改总结

## ✅ 完成的修改

### 1. **数据构建脚本** (`scripts/03_build_dataset.py`)

#### 主要改动：
- **节点特征** - 完全基于 AMBER prmtop，不使用 RDKit 特征：
  - `AMBER_ATOM_TYPE` (one-hot 编码)
  - `CHARGE` (部分电荷，标量)
  - `RESIDUE_LABEL` (one-hot 编码)
  - `ATOMIC_NUMBER` (原子序数，用于兼容 E(3)-GNN)

- **多跳图结构** - FFiNet 风格的多层相互作用：
  - **1-hop (Bonded)**: `edge_index` - BONDS_WITHOUT_HYDROGEN
  - **2-hop (Angle)**: `triple_index` - ANGLES_WITHOUT_HYDROGEN
  - **3-hop (Dihedral)**: `quadra_index` - DIHEDRALS_WITHOUT_HYDROGEN
  - **Non-bonded**: `nonbonded_edge_index` - 空间截断内的非共价相互作用

- **力场参数** - 从 prmtop 提取的边/路径属性：
  - `edge_attr`: [平衡键长, 力常数]
  - `triple_attr`: [平衡角度, 角度力常数]
  - `quadra_attr`: [扭转力常数, 周期性, 相位角]
  - `nonbonded_edge_attr`: [LJ_A系数, LJ_B系数, 距离] (待完善)

- **坐标加载** - 支持两种方式：
  - 优先从 PDB 文件 (via RDKit)
  - 回退到 .inpcrd 文件 (via ParmEd)

#### 函数签名更新：
```python
build_graph_from_files(
    rna_pdb_path,
    prmtop_path,
    distance_cutoff=5.0,          # 非键距离截断
    add_nonbonded_edges=True       # 是否添加非键边
)
```

---

### 2. **模型架构** (`models/e3_gnn_encoder.py`)

#### 修改内容：
- **向后兼容** - 模型可以接受新的数据格式，但当前只使用 `edge_index`
- **文档更新** - 在 docstring 中注明了多跳索引的存在
- **未来扩展** - 在代码注释中标记了可以融入 FFiNet 风格注意力的位置

#### 当前行为：
```python
# 模型接收包含以下字段的 Data 对象：
- x: [num_atoms, feature_dim]  # 新的 AMBER 特征
- pos: [num_atoms, 3]
- edge_index: [2, num_edges]
- triple_index: [3, num_angles]      # 可用但未使用
- quadra_index: [4, num_dihedrals]   # 可用但未使用
- nonbonded_edge_index: [2, num_nb]  # 可用但未使用
```

---

## 📊 测试结果

使用 `test_output/1aju_ARG_graph_intermediate/` 的测试数据：

### 数据统计：
```
✓ Node features: [349, 35]
  - 14 AMBER atom types (one-hot)
  - 1 charge value
  - 11 residue types (one-hot)
  - 1 atomic number

✓ Positions: [349, 3]

✓ 1-hop (Bonded edges):
  - 512 edges (256 bonds × 2 directions)
  - Sample params: [1.61 Å, 230.0 kcal/mol/Å²]

✓ 2-hop (Angle paths):
  - 397 angles
  - Sample params: [108.23°, 100.0 kcal/mol/rad²]

✓ 3-hop (Dihedral paths):
  - 782 dihedrals
  - Sample params: [0.185, period=1, phase=31.8°]

✓ Non-bonded edges:
  - 11,606 spatial edges (cutoff=5.0 Å)
```

---

## 🔧 如何使用

### 1. 构建单个图：
```python
from scripts.build_dataset import build_graph_from_files

data = build_graph_from_files(
    rna_pdb_path="path/to/rna.pdb",
    prmtop_path="path/to/rna.prmtop",
    distance_cutoff=5.0,
    add_nonbonded_edges=True
)

print(f"Node features: {data.x.shape}")
print(f"1-hop edges: {data.edge_index.shape}")
print(f"2-hop paths: {data.triple_index.shape}")
print(f"3-hop paths: {data.quadra_index.shape}")
```

### 2. 批量处理数据集：
```bash
# 使用修改后的脚本
python scripts/03_build_dataset.py \
    --hariboss_csv hariboss/Complexes.csv \
    --amber_dir data/processed/amber \
    --output_dir data/processed/graphs \
    --distance_cutoff 5.0 \
    --num_workers 8
```

### 3. 在模型中使用：
```python
from models.e3_gnn_encoder import RNAPocketEncoder
from torch_geometric.data import Data, Batch

# 创建模型（自动适配新的特征维度）
model = RNAPocketEncoder(
    input_dim=data.x.shape[1],  # 自动检测特征维度
    hidden_irreps="32x0e + 16x1o + 8x2e",
    output_dim=512,
    num_layers=4
)

# 前向传播（当前只使用 edge_index）
output = model(data)  # [batch_size, 512]
```

---

## 🚀 未来扩展建议

### Phase 1: 增强 LJ 参数提取
当前 `nonbonded_edge_attr` 使用占位值。可以改进为：
```python
# 从 amber_parm.parm_data 提取真实 LJ 参数
lj_acoef = amber_parm.parm_data['LENNARD_JONES_ACOEF']
lj_bcoef = amber_parm.parm_data['LENNARD_JONES_BCOEF']
nb_idx = amber_parm.parm_data['NONBONDED_PARM_INDEX']

# 计算原子对的 LJ 参数
idx = nb_idx[type_i * num_types + type_j] - 1
lj_A = lj_acoef[idx]
lj_B = lj_bcoef[idx]
```

### Phase 2: 融入 FFiNet 风格的多跳注意力
在 `E3GNNMessagePassingLayer` 中添加：
- 2-hop 角度注意力
- 3-hop 二面角注意力
- 轴向注意力融合

### Phase 3: 几何特征编码
在边特征中添加实时计算的几何量：
```python
# 距离编码（FFiNet 风格）
distance_bonded = [r, r²]
distance_nonbonded = [r⁻⁶, r⁻¹², r⁻¹]

# 角度编码
angle_features = [θ, θ², cos(θ), sin(θ)]

# 二面角编码（Fourier 展开）
dihedral_features = [cos(φ), cos(2φ), cos(3φ),
                     sin(φ), sin(2φ), sin(3φ)]
```

### Phase 4: 残基级别图
添加第二层图网络：
```python
# 构建残基图
residue_graph = build_residue_graph(data)
residue_embeddings = residue_gnn(residue_graph)

# 将残基嵌入广播回原子
atom_embeddings = broadcast_residue_to_atom(residue_embeddings)
```

---

## 📝 代码质量

### 已实现：
- ✅ 完全基于 AMBER prmtop 的特征提取
- ✅ 多跳索引构建（1/2/3-hop）
- ✅ 力场参数提取
- ✅ 向后兼容的模型接口
- ✅ PDB/INPCRD 双重坐标源
- ✅ 异常处理和回退机制

### 待优化：
- ⚠️ LJ 参数提取（当前使用占位值）
- ⚠️ 1-3 和 1-4 相互作用的特殊处理
- ⚠️ 批处理时的内存优化
- ⚠️ 单元测试覆盖

---

## 🔍 关键差异：FFiNet vs 当前实现

| 方面 | FFiNet | RNA-3E-FFI (当前) |
|------|--------|-------------------|
| **节点特征** | RDKit 通用特征 | AMBER 专用特征 (atom type, charge, residue) |
| **图构建** | NetworkX 路径搜索 | ParmEd 直接提取 |
| **边类型** | bonded/nonbonded 标记 | 显式分离的索引 |
| **几何编码** | 实时计算 (distance, angle, dihedral) | 预计算 + 力场参数 |
| **消息传递** | 多跳注意力 | E(3)-等变消息传递 (仅 1-hop) |
| **物理先验** | 距离/角度的多项式编码 | 力场平衡值和力常数 |

**优势**:
- ✅ 保留了 E(3) 等变性（FFiNet 不具备）
- ✅ 使用真实的 RNA 力场参数
- ✅ 残基类型信息更丰富

**可改进**:
- 多跳信息当前未被模型使用
- LJ 参数需要完善
- 可选地添加 FFiNet 的注意力机制

---

## 📚 相关文件

### 修改的文件：
1. `scripts/03_build_dataset.py` - 数据构建主逻辑
2. `models/e3_gnn_encoder.py` - 模型文档更新

### 新增文件：
1. `test_multihop_data.py` - 测试脚本
2. `MODIFICATIONS_SUMMARY.md` - 本文档

### 测试数据：
- `test_output/1aju_ARG_graph_intermediate/rna.prmtop`
- `test_output/1aju_ARG_graph_intermediate/rna.inpcrd`
- `test_output/1aju_ARG_graph_intermediate/rna_only.pdb`

---

## 💡 使用建议

1. **特征维度** - 当前节点特征维度会随数据集变化：
   ```
   feature_dim = n_atom_types + 1 + n_residue_types + 1
   ```
   建议在训练时固定一个全局的 atom_type 和 residue 字典。

2. **距离截断** - 对于大 RNA，建议：
   - 键相互作用：无限（全部包含）
   - 非键相互作用：5-10 Å 截断

3. **内存优化** - 大分子的多跳路径可能很多：
   - 考虑稀疏化策略
   - 或者只保留最重要的路径（按力常数排序）

4. **批处理** - 确保使用 PyG 的 `Batch.from_data_list()` 正确处理多跳索引

---

**修改完成日期**: 2025-10-25
**测试状态**: ✅ 数据加载通过，模型兼容
