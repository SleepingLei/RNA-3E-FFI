# 受体分类脚本使用说明

## 脚本功能

`classify_receptors.py` 脚本用于根据受体组成（RNA、蛋白质或复合物）对受体文件进行分类和复制。

### 主要特性

1. **自动匹配**：根据配体文件名自动匹配对应的受体文件
2. **智能分类**：分析受体组成，分类为 RNA、蛋白质或复合物
3. **多进程处理**：使用 multiprocessing 加速处理
4. **详细统计**：生成完整的分类报告和统计信息
5. **批量处理**：支持处理所有 processed_ligands_effect_* 目录

## 本地测试结果

在本地示例数据上的测试结果：

| 分类 | 数量 | 百分比 | 示例 |
|-----|------|--------|-----|
| RNA | 1 | 33.33% | 100D-assembly1 (C, DC, DG, G) |
| Protein | 2 | 66.67% | 101M-assembly1, 102L-assembly1 |
| Complex | 0 | 0% | - |

## 文件匹配规则

脚本根据文件名进行匹配：

- **配体文件**：`<PDB_ID>_ligands.pdb`（例如：`100D-assembly1_ligands.pdb`）
- **受体文件**：`<PDB_ID>_polymer_fixed.pdb`（例如：`100D-assembly1_polymer_fixed.pdb`）

提取 PDB ID 后查找对应的受体文件。

## 分类标准

### RNA 识别

包含以下残基类型之一：
- 标准核苷酸：A, U, G, C, T
- DNA 核苷酸：DA, DT, DG, DC
- 修饰核苷酸：PSU, 5MU, 5MC, 7MG, M2G, 1MA 等

### 蛋白质识别

包含以下残基类型之一：
- 20 种标准氨基酸：ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL
- 组氨酸变体：HID, HIE, HIP
- 其他变体：CYX, CYM, ASH, GLH, LYN, MSE

### 复合物识别

同时包含 RNA 和蛋白质残基类型。

## 使用方法

### 方法1：使用快速脚本（推荐）

```bash
# 处理单个目录（processed_ligands_effect_1）
./run_classification.sh

# 处理所有 processed_ligands_effect_* 目录
./run_classification.sh --all

# 自定义参数
./run_classification.sh \
  --ligand-dir processed_ligands_effect_2 \
  --receptor-dir processed_polymers_fixed \
  --output-dir my_output \
  --workers 8
```

### 方法2：直接运行 Python 脚本

```bash
# 处理单个目录
python classify_receptors.py

# 处理所有目录
python classify_receptors.py --all-ligand-dirs

# 自定义参数
python classify_receptors.py \
  --ligand-dir processed_ligands_effect_1 \
  --receptor-dir processed_polymers_fixed \
  --output-dir effect_receptor \
  --workers 8
```

## 参数说明

### Python 脚本参数

- `--ligand-dir`：包含配体 PDB 文件的目录（默认：processed_ligands_effect_1）
- `--receptor-dir`：包含受体 PDB 文件的目录（默认：processed_polymers_fixed）
- `--output-dir`：输出基础目录（默认：effect_receptor）
- `--workers`：并行处理的工作进程数（默认：CPU 核心数）
- `--all-ligand-dirs`：处理所有 processed_ligands_effect_* 目录

### Shell 脚本参数

- `--all`：处理所有 processed_ligands_effect_* 目录
- `--ligand-dir DIR`：指定配体目录
- `--receptor-dir DIR`：指定受体目录
- `--output-dir DIR`：指定输出目录
- `--workers N`：指定工作进程数

## 输出结构

脚本会在输出目录下创建以下子文件夹：

```
effect_receptor/
├── RNA/                          # RNA 受体文件
│   └── *_polymer_fixed.pdb
├── Protein/                      # 蛋白质受体文件
│   └── *_polymer_fixed.pdb
├── Complex/                      # RNA-蛋白质复合物文件
│   └── *_polymer_fixed.pdb
├── Unknown/                      # 无法分类的文件（如果有）
│   └── *_polymer_fixed.pdb
└── classification_report.txt    # 详细分类报告
```

## 在远程机器上运行

### 步骤1：上传文件

```bash
scp classify_receptors.py user@remote:/path/to/dir/
scp run_classification.sh user@remote:/path/to/dir/
```

### 步骤2：准备数据

确保远程机器上有以下目录：
- `processed_ligands_effect_*/`（包含配体 PDB 文件）
- `processed_polymers_fixed/`（包含受体 PDB 文件）

### 步骤3：运行脚本

```bash
# SSH 登录
ssh user@remote
cd /path/to/dir/

# 处理所有配体目录（推荐）
nohup ./run_classification.sh --all > classification.log 2>&1 &

# 或使用 screen
screen -S classification
./run_classification.sh --all
# Ctrl+A+D 分离会话
```

### 步骤4：监控进度

```bash
# 查看日志
tail -f classification.log

# 或重新连接到 screen
screen -r classification
```

### 步骤5：查看结果

```bash
# 查看生成的文件夹
ls -la effect_receptor/

# 统计各类型数量
echo "RNA: $(ls -1 effect_receptor/RNA/*.pdb 2>/dev/null | wc -l)"
echo "Protein: $(ls -1 effect_receptor/Protein/*.pdb 2>/dev/null | wc -l)"
echo "Complex: $(ls -1 effect_receptor/Complex/*.pdb 2>/dev/null | wc -l)"

# 查看详细报告
cat effect_receptor/classification_report.txt
```

## 性能优化建议

### 工作进程数选择

- **推荐设置**：CPU 核心数
- **高 I/O 场景**：可设置为核心数的 1.5-2 倍
- **内存受限**：减少工作进程数

示例：
```bash
# 查看 CPU 核心数
python -c "import multiprocessing; print(multiprocessing.cpu_count())"

# 使用所有核心
./run_classification.sh --all --workers $(nproc)
```

## 分类报告说明

脚本生成的 `classification_report.txt` 包含：

1. **总体统计**：
   - 处理的配体文件总数
   - 成功分类的文件数
   - 未找到受体的文件数
   - 错误数

2. **详细结果**：
   - 每个分类的所有文件
   - 每个文件包含的残基类型

3. **汇总统计**：
   - 各分类的数量和百分比

## 常见问题

### 问题1：找不到受体文件

```
Warning: Receptor not found for XXX-assembly1
```

**原因**：
- 配体文件有对应记录，但受体文件不存在
- 文件命名不匹配

**解决方法**：
- 检查 `processed_polymers_fixed/` 目录中是否存在对应的受体文件
- 确认文件命名格式正确

### 问题2：分类为 Unknown

**原因**：
- 受体文件包含的残基类型不在预定义列表中
- 可能是非标准修饰残基

**解决方法**：
- 查看报告中的残基列表
- 如果是已知的 RNA 或蛋白质残基，可以编辑脚本添加到对应列表
- 修改 `RNA_RESIDUES` 或 `PROTEIN_RESIDUES` 集合

### 问题3：处理速度慢

**解决方法**：
- 增加 `--workers` 参数
- 确保输入/输出目录在快速存储设备上（如 SSD）
- 检查是否有其他进程占用 CPU/磁盘

### 问题4：内存不足

**解决方法**：
- 减少 `--workers` 参数
- 分批处理不同的 processed_ligands_effect_* 目录

## 使用示例

### 示例1：处理单个配体目录

```bash
# 只处理 processed_ligands_effect_1
python classify_receptors.py \
  --ligand-dir processed_ligands_effect_1 \
  --output-dir receptors_effect_1
```

### 示例2：处理所有配体目录

```bash
# 处理所有 processed_ligands_effect_* 目录
python classify_receptors.py --all-ligand-dirs
```

### 示例3：使用多个工作进程

```bash
# 使用 16 个工作进程
./run_classification.sh --all --workers 16
```

### 示例4：自定义所有参数

```bash
python classify_receptors.py \
  --ligand-dir my_ligands \
  --receptor-dir my_receptors \
  --output-dir my_output \
  --workers 8
```

## 输出示例

```
Found 3 ligand files to process
Receptor directory: processed_polymers_fixed
Output directory: effect_receptor

Processing with 2 worker processes...

================================================================================
CLASSIFICATION COMPLETE
================================================================================

Total ligand files processed: 3
Successfully classified: 3
Receptor not found: 0
Errors: 0

--------------------------------------------------------------------------------
CLASSIFICATION SUMMARY
--------------------------------------------------------------------------------

RNA: 1 files
  Output directory: effect_receptor/RNA/
    - 100D-assembly1: C, DC, DG, G

Protein: 2 files
  Output directory: effect_receptor/Protein/
    - 101M-assembly1: ALA, ARG, ASN, ASP, GLN, GLU, GLY, HID, HIE, ILE, ...
    - 102L-assembly1: ALA, ARG, ASN, ASP, GLN, GLU, GLY, HID, ILE, LEU, ...

--------------------------------------------------------------------------------
SUMMARY TABLE
--------------------------------------------------------------------------------
Classification       Count           Percentage
--------------------------------------------------------------------------------
RNA                  1                33.33%
Protein              2                66.67%
--------------------------------------------------------------------------------
Total                3               100.00%
--------------------------------------------------------------------------------

Detailed report saved to: effect_receptor/classification_report.txt

Done!
```

## 扩展残基类型

如果需要添加新的残基类型到识别列表，编辑 `classify_receptors.py` 文件：

```python
# 添加 RNA 残基
RNA_RESIDUES = {
    'A', 'U', 'G', 'C', 'T',
    'DA', 'DT', 'DG', 'DC',
    # 在这里添加新的 RNA 残基
    'YOUR_NEW_RNA_RESIDUE',
}

# 添加蛋白质残基
PROTEIN_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    # ... 其他标准氨基酸
    # 在这里添加新的蛋白质残基
    'YOUR_NEW_PROTEIN_RESIDUE',
}
```

## 注意事项

1. **文件覆盖**：如果输出目录已存在同名文件，会被覆盖
2. **磁盘空间**：确保有足够的磁盘空间存储复制的受体文件
3. **备份**：建议在运行前备份原始数据
4. **命名规范**：确保文件命名遵循标准格式
5. **并行处理**：多个进程会同时访问源文件，确保文件系统支持

## 工作流程整合

典型的数据处理工作流程：

```bash
# 1. 提取有效配体
python extract_effective_ligands.py --workers 16

# 2. 分类受体
python classify_receptors.py --all-ligand-dirs --workers 16

# 3. 查看结果
cat ligand_extraction_report.txt
cat effect_receptor/classification_report.txt
```

或使用快速脚本：

```bash
# 1. 提取有效配体
./run_extraction.sh

# 2. 分类受体
./run_classification.sh --all

# 3. 查看结果
ls -la processed_ligands_effect_*/
ls -la effect_receptor/*/
```

## 联系方式

如有问题或建议，请联系开发者。
