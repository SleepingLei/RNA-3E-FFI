# Generate Pocket Graph for Virtual Screening

这个脚本可以从口袋的PDB或MOL2文件生成用于虚拟筛选的分子图。

## 功能特点

生成的graph与`03_build_dataset.py`完全兼容,包含:
- **节点特征**: AMBER原子类型、电荷、残基类型、原子序数 (使用`amber_vocabulary`)
- **1-hop边**: 共价键 (不含氢原子)
- **2-hop路径**: 角度相互作用 (angle paths)
- **3-hop路径**: 二面角相互作用 (dihedral paths)
- **非键合边**: 基于空间距离的Lennard-Jones参数

## 使用方法

### 基本用法

```bash
# 从PDB文件生成graph (自动检测分子类型)
python scripts/generate_pocket_graph.py --input pocket.pdb --output pocket_graph.pt

# 从MOL2文件生成graph
python scripts/generate_pocket_graph.py --input pocket.mol2 --output pocket_graph.pt
```

### 高级选项

```bash
# **推荐**: 使用归一化参数进行模型推理
python scripts/generate_pocket_graph.py \
    --input pocket.pdb \
    --output pocket_graph.pt \
    --norm_params data/processed/node_feature_norm_params.npz

# 明确指定分子类型 (rna/protein/mixed)
python scripts/generate_pocket_graph.py \
    --input pocket.pdb \
    --output pocket_graph.pt \
    --molecule_type rna

# 自定义非键合边的距离cutoff (默认5.0 Å)
python scripts/generate_pocket_graph.py \
    --input pocket.pdb \
    --output pocket_graph.pt \
    --distance_cutoff 6.0

# 跳过非键合边构建 (加快速度)
python scripts/generate_pocket_graph.py \
    --input pocket.pdb \
    --output pocket_graph.pt \
    --no_nonbonded_edges

# 保留中间AMBER文件用于调试
python scripts/generate_pocket_graph.py \
    --input pocket.pdb \
    --output pocket_graph.pt \
    --keep_temp
```

## 输入格式

支持两种输入格式:
- **PDB格式** (.pdb): 蛋白质数据库格式
- **MOL2格式** (.mol2): Sybyl MOL2格式

输入文件可以包含:
- 纯RNA口袋
- 纯蛋白质口袋
- RNA-蛋白质混合口袋

脚本会自动检测分子类型并选择合适的AMBER力场:
- RNA: `leaprc.RNA.OL3`
- Protein: `leaprc.protein.ff14SB`
- Mixed: 两者都加载

## 输出格式

输出是PyTorch Geometric的Data对象 (.pt文件),包含:

```python
data = torch.load("pocket_graph.pt")

# 节点特征和位置
data.x                      # [N, 4] 节点特征矩阵
data.pos                    # [N, 3] 3D坐标

# 1-hop: 化学键
data.edge_index            # [2, E1] 边索引
data.edge_attr             # [E1, 2] 键参数 [平衡长度, 力常数]

# 2-hop: 角度路径
data.triple_index          # [3, E2] 角度路径 [i, j, k]
data.triple_attr           # [E2, 2] 角度参数 [平衡角, 力常数]

# 3-hop: 二面角路径
data.quadra_index          # [4, E3] 二面角路径 [i, j, k, l]
data.quadra_attr           # [E3, 3] 二面角参数 [势垒, 周期性, 相位]

# 非键合边
data.nonbonded_edge_index  # [2, E4] 空间邻近边
data.nonbonded_edge_attr   # [E4, 3] LJ参数 [log(A), log(B), 距离]
```

## 依赖要求

需要安装以下软件:
- **AmberTools**: tleap用于参数化
- **Python包**:
  - torch
  - torch_geometric
  - parmed
  - MDAnalysis
  - numpy

## 工作流程

1. **分子类型检测** (可选): 自动识别RNA/protein/mixed
2. **格式转换**: MOL2转PDB (如果需要)
3. **末端清理**: 清理RNA片段的末端原子 (tleap会重新添加)
4. **AMBER参数化**: 使用tleap生成prmtop和inpcrd文件
5. **图构建**: 从AMBER拓扑提取节点、边、路径信息
6. **保存**: 输出PyTorch Geometric格式的.pt文件

## 示例: 从现有口袋生成graph

假设你有一个口袋PDB文件 `data/processed/pockets/1aju_ARG_model0_pocket.pdb`:

```bash
# 推荐方式：使用归一化参数（用于模型推理）
python scripts/generate_pocket_graph.py \
    --input data/processed/pockets/1aju_ARG_model0_pocket.pdb \
    --output virtual_screening/1aju_pocket_graph.pt \
    --molecule_type rna \
    --distance_cutoff 5.0 \
    --norm_params data/processed/node_feature_norm_params.npz

# 不使用归一化（仅用于探索性分析）
python scripts/generate_pocket_graph.py \
    --input data/processed/pockets/1aju_ARG_model0_pocket.pdb \
    --output virtual_screening/1aju_pocket_graph.pt \
    --molecule_type rna \
    --distance_cutoff 5.0
```

输出文件可以直接用于虚拟筛选模型推理。

## 调试

如果遇到问题:

1. **保留中间文件**: 使用`--keep_temp`查看AMBER参数化的详细信息
2. **检查分子类型**: 使用`--molecule_type`明确指定类型
3. **简化graph**: 使用`--no_nonbonded_edges`跳过非键合边

## 与数据集构建的区别

| 特性 | `03_build_dataset.py` | `generate_pocket_graph.py` |
|------|----------------------|----------------------------|
| 输入 | 已处理的HARIBOSS数据集 | 单个口袋PDB/MOL2文件 |
| 批量处理 | 多进程处理大量复合物 | 单个文件处理 |
| 归一化 | 全局特征归一化 | 支持归一化 (通过`--norm_params`) |
| 用途 | 训练数据集构建 | 虚拟筛选推理 |
| graph格式 | 完全相同 | 完全相同 |

## 节点特征归一化

**重要**: 如果你的模型是在归一化数据上训练的，在推理时必须使用相同的归一化参数！

### 归一化参数文件

归一化参数文件 `data/processed/node_feature_norm_params.npz` 包含:
- `mean`: 训练集的特征均值 (4个特征)
- `std`: 训练集的特征标准差 (4个特征)
- `continuous_indices`: 需要归一化的特征索引 `[1, 3]`
  - 索引1: 原子电荷 (charge)
  - 索引3: 原子序数 (atomic_number)

### 使用归一化

```bash
# 推荐用法：使用归一化参数
python scripts/generate_pocket_graph.py \
    --input pocket.pdb \
    --output pocket_graph.pt \
    --norm_params data/processed/node_feature_norm_params.npz
```

### 归一化公式

对于连续特征（电荷和原子序数）:
```
normalized_value = (original_value - mean) / std
```

### 查看归一化参数

```python
import numpy as np
data = np.load('data/processed/node_feature_norm_params.npz')
print("Mean:", data['mean'])
print("Std:", data['std'])
print("Continuous indices:", data['continuous_indices'])
```

输出:
```
Mean: [ 0.         -0.0283311   0.          5.09101152]
Std: [1.         0.46654615 1.         3.31272888]
Continuous indices: [1 3]
```

## 注意事项

1. **特征归一化**:
   - **模型推理必须使用归一化**: 通过 `--norm_params` 参数指定归一化文件
   - **不归一化会导致预测错误**: 特征分布不匹配会严重影响模型性能
   - **使用训练集的参数**: 归一化参数来自训练数据集，不要用新数据重新计算

2. **配体分离**: 此脚本处理整个口袋,不会自动分离配体

3. **AMBER环境**: 需要正确配置AmberTools环境
