# RNA-3E-FFI 数据处理流程

## 概述

完整的数据处理流程，将受体和配体转换为可用于模型训练的格式：
- 受体 → AMBER 参数化 → 分子图
- 配体 → SMILES → Uni-Mol embedding

## 文件说明

### 核心脚本

| 脚本 | 功能 |
|-----|------|
| `parameterize_receptors.py` | AMBER 参数化受体（支持 RNA/DNA/Protein/Complex） |
| `build_receptor_graphs.py` | 从 AMBER 文件构建分子图 |
| `pdb_to_smiles.py` | 将配体 PDB 转换为 SMILES（pH 7.4 校正） |
| `generate_ligand_embeddings.py` | 从 SMILES 生成 Uni-Mol embedding |
| `run_full_pipeline.sh` | 完整流程脚本 |

## 快速开始

```bash
# 完整流程（一键运行）
./run_full_pipeline.sh
```

## 详细步骤

### 步骤 1: 转换配体为 SMILES

```bash
python pdb_to_smiles.py \
    --input-dir processed_ligands_effect_1 \
    --output-csv ligands_smiles.csv \
    --ph 7.4 \
    --workers 16
```

**输出**: `ligands_smiles.csv`（包含 filename, molecule_name, smiles, corrected_smiles）

### 步骤 2: 参数化受体

```bash
python parameterize_receptors.py \
    --receptor-dir effect_receptor \
    --output-dir processed_output \
    --workers 16
```

**功能**:
- 根据受体类型自动选择力场：
  - RNA/DNA → RNA.OL3
  - Protein → ff14SB
  - Complex → RNA.OL3 + ff14SB
- 生成 .prmtop 和 .inpcrd 文件

**输出**: `processed_output/amber/*.prmtop`, `*.inpcrd`

### 步骤 3: 构建受体图

```bash
python build_receptor_graphs.py \
    --amber-dir processed_output/amber \
    --output-dir processed_output \
    --distance-cutoff 5.0 \
    --workers 16
```

**功能**:
- 节点特征: [charge, atomic_number, mass]
- 1-hop: 共价键
- 2-hop: 角度
- 3-hop: 二面角
- Non-bonded: Lennard-Jones 参数

**输出**: `processed_output/graphs/*.pt`（PyTorch Geometric格式）

### 步骤 4: 生成配体 embedding

```bash
python generate_ligand_embeddings.py \
    --csv-file ligands_smiles.csv \
    --output-h5 processed_output/ligand_embeddings.h5 \
    --batch-size 32
```

**功能**:
- 使用 Uni-Mol2 生成 embedding
- Z-score 归一化
- 保存归一化参数

**输出**:
- `processed_output/ligand_embeddings.h5`
- `processed_output/ligand_embedding_norm_params.npz`

## 依赖环境

### AMBER Tools
```bash
# 需要安装 AmberTools
conda install -c conda-forge ambertools
```

### Python 包
```bash
pip install numpy pandas torch torch_geometric
pip install MDAnalysis parmed
pip install openbabel
pip install unimol_tools h5py
```

## 输出结构

```
processed_output/
├── amber/
│   ├── {pdb_id}.prmtop
│   └── {pdb_id}.inpcrd
├── graphs/
│   └── {pdb_id}.pt
├── ligand_embeddings.h5
├── ligand_embedding_norm_params.npz
└── graph_summary.pkl
```

## 性能建议

| 文件数 | 推荐 Workers | 预计时间 |
|-------|-------------|----------|
| < 10 | 4 | < 5分钟 |
| 10-100 | 8-16 | 5-30分钟 |
| > 100 | 16-32 | 30分钟+ |

## 故障排除

### 问题 1: AmberTools 未安装
```
Error: tleap: command not found
```
解决: `conda install -c conda-forge ambertools`

### 问题 2: OpenBabel 未安装
```
Error: obabel: command not found
```
解决: `conda install -c conda-forge openbabel`

### 问题 3: Uni-Mol 未安装
```
Error: unimol_tools not installed
```
解决: `pip install unimol_tools`

### 问题 4: CUDA 内存不足
减少 batch_size:
```bash
python generate_ligand_embeddings.py --batch-size 8
```

## 数据验证

### 验证图文件
```python
import torch
graph = torch.load('processed_output/graphs/100D-assembly1.pt')
print(f"Nodes: {graph.x.shape}")
print(f"Edges: {graph.edge_index.shape}")
```

### 验证 embedding
```python
import h5py
with h5py.File('processed_output/ligand_embeddings.h5', 'r') as f:
    print(f"Keys: {list(f.keys())}")
    emb = f['100D-assembly1'][:]
    print(f"Shape: {emb.shape}")
```

## 批量处理

### 处理多个配体目录
```bash
for dir in processed_ligands_effect_*/; do
    python pdb_to_smiles.py --input-dir $dir --output-csv ${dir%/}_smiles.csv
done
```

### 远程服务器运行
```bash
# 使用 nohup
nohup ./run_full_pipeline.sh > pipeline.log 2>&1 &

# 或使用 screen
screen -S pipeline
./run_full_pipeline.sh
# Ctrl+A+D 分离
```

## 注意事项

1. **内存使用**: 大型受体可能需要大量内存
2. **GPU**: Uni-Mol embedding 生成建议使用 GPU
3. **磁盘空间**: 确保有足够空间存储中间文件
4. **AMBER 许可**: AmberTools 免费，但完整 AMBER 需要许可

## 引用

如果使用此流程，请引用：
- AMBER: http://ambermd.org
- Uni-Mol: https://github.com/dptech-corp/Uni-Mol
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io
