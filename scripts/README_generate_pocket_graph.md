# Generate Pocket Graph Script

## 概述

`generate_pocket_graph.py` 脚本用于从给定的RNA结合口袋生成模型所需的PyTorch Geometric图数据。

该脚本完成以下步骤:
1. **提取RNA**: 从口袋PDB文件中提取RNA残基(排除配体、蛋白质等)
2. **参数化**: 使用Amber RNA.OL3力场对RNA进行参数化
3. **构建图**: 生成包含节点特征、边和3D坐标的分子图

## 功能特点

- 自动从口袋中分离RNA和配体
- 支持标准RNA残基和修饰RNA残基
- 使用Amber力场进行参数化(tleap)
- 结合RDKit和ParmEd提取丰富的原子特征
- 基于距离阈值构建边
- 输出PyTorch Geometric Data格式

## 节点特征 (11维)

每个原子节点包含以下特征:
1. **原子序数** (1维): 原子的原子序数
2. **杂化类型** (5维): one-hot编码 [SP, SP2, SP3, SP3D, SP3D2]
3. **芳香性** (1维): 是否为芳香原子
4. **度数** (1维): 原子的连接度
5. **形式电荷** (1维): 原子的形式电荷
6. **部分电荷** (1维): Amber力场计算的部分电荷
7. **原子类型哈希** (1维): Amber原子类型的哈希值

## 使用方法

### 基本用法

```bash
python scripts/generate_pocket_graph.py \
    --input pocket.pdb \
    --output pocket_graph.pt
```

### 排除配体

如果输入PDB包含配体,需要指定配体残基名:

```bash
python scripts/generate_pocket_graph.py \
    --input data/processed/pockets/1aju_ARG_pocket.pdb \
    --output graphs/1aju_ARG_graph.pt \
    --ligand_resname ARG
```

### 自定义距离阈值

调整边构建的距离阈值(默认4.0Å):

```bash
python scripts/generate_pocket_graph.py \
    --input pocket.pdb \
    --output pocket_graph.pt \
    --distance_cutoff 5.0
```

### 保留中间文件(调试用)

```bash
python scripts/generate_pocket_graph.py \
    --input pocket.pdb \
    --output pocket_graph.pt \
    --keep_intermediate
```

## 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--input` | str | 是 | - | 输入PDB文件(RNA结合口袋) |
| `--output` | str | 是 | - | 输出图文件(.pt) |
| `--ligand_resname` | str | 否 | None | 要从RNA中排除的配体残基名 |
| `--distance_cutoff` | float | 否 | 4.0 | 边构建的距离阈值(Å) |
| `--keep_intermediate` | flag | 否 | False | 保留中间文件用于调试 |

## 输出格式

生成的`.pt`文件包含一个PyTorch Geometric `Data`对象:

```python
import torch
data = torch.load('pocket_graph.pt')

# 数据结构
print(data)
# Data(x=[N, 11], edge_index=[2, E], pos=[N, 3])

# 其中:
# - x: 节点特征矩阵 [节点数, 11维特征]
# - edge_index: 边索引 [2, 边数]
# - pos: 节点3D坐标 [节点数, 3]
```

## 示例

### 示例1: 处理单个口袋

```bash
python scripts/generate_pocket_graph.py \
    --input data/processed/pockets/1aju_ARG_pocket.pdb \
    --output graphs/1aju_ARG_graph.pt \
    --ligand_resname ARG
```

输出:
```
======================================================================
Generating graph from RNA binding pocket
======================================================================
Input: data/processed/pockets/1aju_ARG_pocket.pdb
Output: graphs/1aju_ARG_graph.pt
Excluding ligand: ARG
Distance cutoff: 4.0Å
======================================================================

Found 350 RNA atoms in 11 residues
✓ Graph construction successful
  - Nodes: 349
  - Node features: 11
  - Edges: 6944
✓ Graph saved to graphs/1aju_ARG_graph.pt
```

### 示例2: 批量处理

```bash
# 批量处理多个口袋
for pocket in data/processed/pockets/*_pocket.pdb; do
    base=$(basename $pocket _pocket.pdb)
    ligand=$(echo $base | cut -d'_' -f2)

    python scripts/generate_pocket_graph.py \
        --input $pocket \
        --output graphs/${base}_graph.pt \
        --ligand_resname $ligand
done
```

## 依赖要求

- Python 3.7+
- PyTorch
- PyTorch Geometric
- RDKit
- ParmEd
- MDAnalysis
- NumPy
- AmberTools (tleap命令)

## 故障排除

### 问题1: tleap未找到

**错误**: `FileNotFoundError: tleap command not found`

**解决**: 安装AmberTools:
```bash
conda install -c conda-forge ambertools
```

### 问题2: 没有找到RNA原子

**错误**: `No RNA atoms found`

**原因**:
- PDB文件中的残基名不在标准列表中
- 所有RNA都被当作配体排除了

**解决**:
- 检查PDB文件中的残基名
- 确认`--ligand_resname`参数正确

### 问题3: 原子数不匹配

**警告**: `Atom count mismatch - RDKit: X, AMBER: Y`

**原因**: terminal cleaning步骤移除了一些原子

**影响**: 通常不影响结果,脚本会使用较小的原子数继续处理

## 技术细节

### RNA残基识别

支持的标准RNA残基:
```python
RNA_RESIDUES = ['A', 'C', 'G', 'U', 'A3', 'A5', 'C3', 'C5',
                'G3', 'G5', 'U3', 'U5', 'DA', 'DC', 'DG', 'DT', ...]
```

支持的修饰RNA残基:
```python
MODIFIED_RNA = ['PSU', '5MU', '5MC', '1MA', '7MG', 'M2G',
                'OMC', 'OMG', 'H2U', '2MG', 'M7G', ...]
```

### 参数化流程

1. 清理末端原子(5'磷酸基团和3'羟基)
2. 使用tleap加载RNA.OL3力场
3. 生成prmtop和inpcrd文件

### 图构建

- 边构建: 欧氏距离 ≤ distance_cutoff
- 无向图: 每条边添加双向
- 自环处理: 如果没有边则添加自环

## 与模型集成

生成的图可以直接用于RNA-3E-FFI模型:

```python
import torch
from torch_geometric.data import Data

# 加载图
graph = torch.load('pocket_graph.pt')

# 用于模型输入
# model = YourRNAModel()
# output = model(graph.x, graph.edge_index, graph.pos)
```

## 相关脚本

- `scripts/01_process_data.py`: 批量处理HARIBOSS数据集
- `scripts/03_build_dataset.py`: 构建训练数据集

## 作者

基于RNA-3E-FFI项目的数据处理流程开发

## 版本历史

- v1.0 (2025-01): 初始版本
  - RNA提取
  - Amber参数化
  - 图构建
  - 中间文件管理
