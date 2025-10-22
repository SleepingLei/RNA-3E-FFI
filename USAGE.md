# 使用说明 | Usage Guide

## 项目目标

**虚拟筛选**：给定一个RNA结合口袋，从配体库中找到潜在的结合配体

**工作流程**：
1. 处理RNA-配体复合物数据 → 提取口袋
2. 生成配体嵌入（Uni-Mol）
3. 训练E(3) GNN将口袋嵌入对齐到配体嵌入空间
4. 虚拟筛选：计算口袋-配体相似度，排序推荐

## 一键运行

```bash
# 1. 检查环境
python scripts/00_check_environment.py

# 2. 处理RNA-配体复合物数据（核心步骤）
python scripts/01_process_data.py \
    --hariboss_csv hariboss/Complexes.csv \
    --output_dir data/processed \
    --pocket_cutoff 5.0 \
    --max_complexes 10

# 3. 生成配体嵌入
python scripts/02_embed_ligands.py \
    --hariboss_csv hariboss/Complexes.csv \
    --output_dir data/processed

# 4. 构建图数据集
python scripts/03_build_dataset.py \
    --processed_dir data/processed \
    --output_dir data/datasets

# 5. 训练模型（对比学习）
python scripts/04_train_model.py \
    --dataset_dir data/datasets \
    --embeddings_path data/processed/ligand_embeddings.h5 \
    --output_dir models

# 6. 虚拟筛选
python scripts/05_run_inference.py \
    --model_path models/best_model.pt \
    --query_pocket data/test/query_pocket.pdb \
    --ligand_library data/processed/ligand_embeddings.h5 \
    --top_k 100
```

## 输出文件说明

每个复合物会生成以下文件：

```
data/processed/
├── pockets/
│   └── 1aju_ARG_pocket.pdb              # 完整口袋（RNA + 配体）
└── amber/
    ├── 1aju_ARG_rna.pdb                 # RNA原始结构
    ├── 1aju_ARG_rna_cleaned.pdb         # 清理后的RNA（末端原子已移除）
    ├── 1aju_ARG_rna.prmtop              # Amber拓扑文件（~150 KB）
    ├── 1aju_ARG_rna.inpcrd              # Amber坐标文件（~12 KB）
    └── 1aju_ARG_ligand_ARG.pdb          # 配体结构
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pocket_cutoff` | 5.0 | 口袋定义的距离阈值（埃） |
| `--max_complexes` | None | 限制处理的复合物数量（测试用） |
| `--output_dir` | data | 输出目录 |
| `--hariboss_csv` | 必需 | HARIBOSS数据集CSV文件路径 |

## 处理结果JSON格式

```json
{
  "pdb_id": "1aju",
  "ligand": "ARG",
  "success": true,
  "components": {
    "rna": {
      "success": true,
      "atoms": 350,
      "residues": 11,
      "prmtop": "data/processed/amber/1aju_ARG_rna.prmtop",
      "inpcrd": "data/processed/amber/1aju_ARG_rna.inpcrd"
    }
  },
  "errors": []
}
```

## 常见问题速查

| 问题 | 解决方案 |
|------|---------|
| NumPy版本错误 | `pip install 'numpy<2.0'` |
| CIF文件未找到 | 确保文件在 `data/raw/mmCIF/` |
| tleap gap警告 | 正常现象（口袋片段） |
| Atom type错误 | 应该已修复，检查脚本版本 |

## 文档索引

- **快速开始**: `docs/QUICK_START.md`
- **技术细节**: `docs/TECHNICAL_SUMMARY.md`
- **项目总览**: `PROJECT_OVERVIEW.md`
- **完整文档**: `README.md`

## 重要提示

⚠️ **归档目录** (`archive/`) 中的文件仅供参考，不需要运行。

✅ **当前版本**的所有核心功能都在 `scripts/01_process_data.py` 中。
