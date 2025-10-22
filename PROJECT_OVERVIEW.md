# RNA-3E-FFI Project Overview | 项目总览

## 项目简介

RNA-3E-FFI 是一个基于E(3)等变图神经网络的RNA-配体虚拟筛选系统。

**目标**：给定RNA结合口袋，从配体库中找到相似的配体
**方法**：通过对比学习将RNA口袋和配体嵌入到共享的潜在空间

**核心创新**：
- ✅ 完整残基选择策略（100% vs 0%完整度）
- ✅ RNA末端原子自动清理
- ✅ 分子分类与独立参数化
- ✅ 鲁棒的错误处理机制
- ✅ E(3)等变口袋编码器 + Uni-Mol配体嵌入

## 项目结构

```
RNA-3E-FFI/
├── scripts/                      # 核心处理脚本
│   ├── 01_process_data.py       # ★ 主要数据处理流程
│   ├── 02_embed_ligands.py      # 配体嵌入生成
│   ├── 03_build_dataset.py      # 图数据集构建
│   ├── 04_train_model.py        # 模型训练
│   └── 05_run_inference.py      # 推理预测
│
├── docs/                         # 文档
│   ├── QUICK_START.md           # 快速开始指南
│   └── TECHNICAL_SUMMARY.md     # 技术总结
│
├── data/                         # 数据目录
│   ├── raw/mmCIF/               # 输入：PDB结构文件
│   └── processed/               # 输出：处理后的数据
│       ├── pockets/             # 口袋结构
│       ├── amber/               # Amber参数文件
│       └── processing_results.json
│
├── archive/                      # 归档（历史版本）
│   ├── old_scripts/             # 旧版本脚本
│   └── old_docs/                # 开发文档
│
├── hariboss/                     # HARIBOSS数据集
│   ├── Complexes.csv            # 复合物元数据
│   └── compounds.csv            # 化合物信息
│
├── README.md                     # 主要README
├── PROJECT_OVERVIEW.md          # 本文件
└── requirements.txt             # Python依赖
```

## 快速开始

### 1. 安装环境

```bash
conda create -n rna-ffi python=3.11
conda activate rna-ffi
pip install -r requirements.txt
pip install 'numpy<2.0'
conda install -c conda-forge ambertools
```

### 2. 处理数据（最重要的步骤）

```bash
python scripts/01_process_data.py \
    --hariboss_csv hariboss/Complexes.csv \
    --output_dir data/processed \
    --pocket_cutoff 5.0 \
    --max_complexes 10
```

### 3. 查看结果

```bash
# 查看处理结果摘要
cat data/processed/processing_results.json

# 查看生成的文件
ls -lh data/processed/amber/
```

## 核心脚本说明

### `scripts/01_process_data.py` ⭐

**最重要的脚本**，实现了所有核心功能：

**主要功能**：
1. 加载RNA-配体复合物结构（mmCIF格式）
2. 分类分子类型（RNA、蛋白质、配体、水、离子）
3. 基于**完整残基**定义结合口袋
4. 清理RNA末端原子
5. 使用Amber RNA.OL3力场参数化

**关键类和函数**：

| 名称 | 功能 | 重要性 |
|------|------|--------|
| `MoleculeClassifier` | 分类所有分子 | ⭐⭐⭐ |
| `define_pocket_by_residues()` | 完整残基选择 | ⭐⭐⭐⭐⭐ |
| `clean_rna_terminal_atoms()` | 清理末端原子 | ⭐⭐⭐⭐⭐ |
| `parameterize_rna()` | RNA参数化 | ⭐⭐⭐⭐ |
| `process_complex_v2()` | 主处理流程 | ⭐⭐⭐⭐ |

**输出文件**：
- `*_pocket.pdb` - 完整口袋结构
- `*_rna.pdb` - 原始RNA
- `*_rna_cleaned.pdb` - 清理后的RNA
- `*_rna.prmtop` - Amber拓扑文件（137-167 KB）
- `*_rna.inpcrd` - Amber坐标文件（10-13 KB）
- `*_ligand_*.pdb` - 配体结构

### `scripts/02_embed_ligands.py`

使用Uni-Mol生成3D配体嵌入向量。

### `scripts/03_build_dataset.py`

从处理后的结构构建PyTorch Geometric图数据集。

### `scripts/04_train_model.py`

训练E(3)等变图神经网络。

### `scripts/05_run_inference.py`

对新的RNA结构进行结合位点预测。

## 关键技术点

### 1. 完整残基选择（核心创新）

**问题**：基于原子选择会导致残基不完整
```python
# ❌ 错误方法
pocket = rna.select_atoms(f"around 5.0 global resname LIG")
# 结果：0%残基完整，pdb4amber崩溃
```

**解决**：基于完整残基选择
```python
# ✅ 正确方法
atoms_in_range = rna.select_atoms(f"around 5.0 global resname LIG")
residues = atoms_in_range.residues
pocket = select_all_atoms_from_residues(residues)
# 结果：100%残基完整
```

**效果**：
- 原子数增加 250-350%
- 残基完整性从 0% → 100%
- 没有pdb4amber崩溃

### 2. 末端原子清理（用户关键洞察）

**用户建议**："这里可以尝试在PDB中删除5'或3'变体，ambertools会自动补全的"

**实现**：
```python
# 移除5'磷酸基团
first_residue: remove P, OP1, OP2

# 移除3'羟基
last_residue: remove O3'

# tleap会用正确的原子类型重新添加
```

**结果**：
- ✅ 无"Atom does not have a type"错误
- ✅ tleap成功生成prmtop/inpcrd文件
- ✅ 仅有预期的gap警告（口袋片段正常现象）

### 3. 分子分类

自动识别：
- **RNA**：A, C, G, U及修饰碱基
- **蛋白质**：20种标准氨基酸
- **配体**：从HARIBOSS元数据提取
- **溶剂和离子**：HOH, NA, CL, MG等

## 测试结果

### 成功率
- 测试复合物：3个
- 成功处理：3个（100%）
- RNA参数化成功：3个（100%）

### 性能
- 平均处理时间：~0.5秒/复合物
- 总处理时间：3个复合物 ~2秒

### 输出文件大小
| PDB ID | prmtop | inpcrd | 残基数 | 原子数 |
|--------|--------|--------|--------|--------|
| 1aju   | 163 KB | 12 KB  | 11     | 350    |
| 1akx   | 137 KB | 10 KB  | 9      | 286    |
| 1am0   | 167 KB | 13 KB  | 11     | 360    |

## 文档导航

### 新用户
1. 阅读 `README.md` - 项目概述
2. 阅读 `docs/QUICK_START.md` - 快速上手
3. 运行示例数据

### 开发者
1. 阅读 `docs/TECHNICAL_SUMMARY.md` - 技术细节
2. 查看 `scripts/01_process_data.py` 源代码
3. 参考 `archive/old_docs/` 中的开发文档（如需了解历史）

### 问题排查
1. 查看 `docs/QUICK_START.md` 的"常见问题"部分
2. 检查 `data/processed/processing_results.json`
3. 查看 `archive/old_docs/` 中的崩溃分析文档

## 归档说明

`archive/` 目录包含：

### `archive/old_scripts/`
- `01_process_data.py` - 旧版V1脚本（基于原子选择）
- `analyze_pdb4amber_crash.py` - pdb4amber崩溃分析
- `debug_*.py` - 各种调试脚本
- `test_pocket_selection.py` - 残基vs原子选择对比测试
- `clean_rna_terminals.py` - 独立的末端清理脚本

### `archive/old_docs/`
- `PDB4AMBER_CRASH_ANALYSIS.md` - pdb4amber崩溃根因分析
- `NEW_PARAMETERIZATION_STRATEGY.md` - 新参数化策略设计
- `POCKET_SELECTION_FIX.md` - 口袋选择修复说明
- `IMPLEMENTATION_STATUS.md` - 开发状态记录

**注意**：归档内容仅供参考，不需要运行。

## 依赖项

### Python包
```
MDAnalysis>=2.0.0
BioPython>=1.79
numpy<2.0  # 重要！pdb4amber需要
pandas>=1.3.0
torch>=2.0.0
torch-geometric>=2.3.0
tqdm>=4.62.0
```

### 外部工具
```
AmberTools (tleap, pdb4amber)
```

## 下一步开发

### 短期任务
- [ ] 实现配体参数化（antechamber + GAFF）
- [ ] 处理修饰RNA残基
- [ ] 添加蛋白质参数化（ff14SB）

### 中期任务
- [ ] 并行处理大数据集
- [ ] 质量控制指标
- [ ] 可视化工具

### 长期任务
- [ ] 支持其他力场（CHARMM, OPLS）
- [ ] 基于ML的口袋验证
- [ ] 与对接工具集成

## 致谢

感谢用户的关键洞察：
1. "选择一定距离中的碱基，后者才比较合理" - 完整残基选择
2. "尝试在PDB中删除5'或3'变体，ambertools会自动补全的" - 末端清理方案

这两个建议是解决所有问题的关键！

## 许可证

[待添加]

## 引用

[待添加]
