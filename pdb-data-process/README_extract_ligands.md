# 提取有效配体小分子脚本使用说明

## 脚本功能

`extract_effective_ligands.py` 脚本用于从 PDB 配体文件中提取有效的小分子，排除常见的辅因子、离子和溶剂分子。

### 主要特性

1. **多进程处理**：使用 multiprocessing 加速处理大量文件
2. **自动分类**：根据每个 PDB 文件中的有效小分子数量自动分类存储
3. **排除列表**：基于 exclude_molecules.txt 文件排除无效分子
4. **详细统计**：生成完整的处理报告和统计信息

## 本地测试结果

在本地示例数据上的测试结果：

- 总文件数：5
- 成功处理：4
- 无有效分子：1
- 错误：0

### 分布情况

- **1个有效分子**：3个文件
  - 100D-assembly1_ligands.pdb: SPM（精胺）
  - 101M-assembly1_ligands.pdb: NBN
  - 102L-assembly1_ligands.pdb: BME

- **2个有效分子**：1个文件
  - 101D-assembly1_ligands.pdb: CBR, NT

- **0个有效分子**：1个文件
  - 104L-assembly1_ligands.pdb（仅包含排除列表中的分子）

## 使用方法

### 基本用法

```bash
python extract_effective_ligands.py
```

这将使用默认参数：
- 输入目录：`processed_ligands/`
- 排除文件：`processed_ligands/exclude_molecules.txt`
- 输出目录：当前目录
- 工作进程数：CPU核心数

### 自定义参数

```bash
python extract_effective_ligands.py \
  --input-dir /path/to/processed_ligands \
  --exclude-file /path/to/exclude_molecules.txt \
  --output-base /path/to/output \
  --workers 8
```

### 参数说明

- `--input-dir`：包含配体 PDB 文件的输入目录（默认：processed_ligands）
- `--exclude-file`：包含需要排除的分子列表的文件（默认：processed_ligands/exclude_molecules.txt）
- `--output-base`：输出文件夹的基础目录（默认：当前目录）
- `--workers`：并行处理的工作进程数（默认：CPU核心数）

## 输出结构

脚本会在输出基础目录下创建以下文件夹：

```
processed_ligands_effect_1/  # 包含1个有效小分子的PDB文件
processed_ligands_effect_2/  # 包含2个有效小分子的PDB文件
processed_ligands_effect_3/  # 包含3个有效小分子的PDB文件
...
ligand_extraction_report.txt # 详细的处理报告
```

## 在远程机器上运行

### 步骤1：上传文件

将以下文件上传到远程机器：
```bash
scp extract_effective_ligands.py user@remote:/path/to/working/dir/
```

### 步骤2：准备数据

确保远程机器上有以下文件和目录：
- `processed_ligands/` 目录（包含所有配体 PDB 文件）
- `processed_ligands/exclude_molecules.txt` 文件

### 步骤3：运行脚本

```bash
# 登录到远程机器
ssh user@remote

# 进入工作目录
cd /path/to/working/dir/

# 运行脚本（推荐使用 nohup 或 screen）
nohup python extract_effective_ligands.py --workers 16 > extraction.log 2>&1 &

# 或使用 screen
screen -S ligand_extraction
python extract_effective_ligands.py --workers 16
# Ctrl+A+D 分离会话
```

### 步骤4：监控进度

```bash
# 查看日志
tail -f extraction.log

# 或重新连接到 screen 会话
screen -r ligand_extraction
```

### 步骤5：查看结果

```bash
# 查看生成的文件夹
ls -la processed_ligands_effect_*/

# 查看详细报告
cat ligand_extraction_report.txt
```

## 性能优化建议

### 工作进程数选择

- **CPU密集型任务**：建议设置为 CPU 核心数
- **I/O密集型任务**：可以设置为 CPU 核心数的 1.5-2 倍
- **内存限制**：如果内存不足，减少工作进程数

示例：
```bash
# 查看 CPU 核心数
python -c "import multiprocessing; print(multiprocessing.cpu_count())"

# 使用所有核心
python extract_effective_ligands.py --workers $(nproc)

# 使用核心数的1.5倍
python extract_effective_ligands.py --workers $(echo "$(nproc)*1.5" | bc | cut -d. -f1)
```

## 排除分子列表

当前排除的分子类型包括：

1. **溶剂分子**：HOH, WAT, EDO, PEG, GOL 等
2. **缓冲剂**：TES, MES, TRS, DTT 等
3. **离子**：NA, K, CA, MG, ZN, CL, FE, MN, CU 等
4. **核苷酸及衍生物**：ATP, ADP, AMP, GTP, GDP, GMP 等
5. **辅因子**：NAD, FAD, COA, HEM 等
6. **其他常见分子**：SO4, PO4, ACE, NME 等

完整列表请查看 `exclude_molecules.txt` 文件。

## 输出文件格式

每个提取的 PDB 文件包含：

```
REMARK   1 EFFECTIVE LIGANDS EXTRACTED
REMARK   1 SOURCE FILE: [原始文件名]
REMARK   1 NUMBER OF EFFECTIVE MOLECULES: [数量]
REMARK   1 MOLECULES: [分子列表]
HETATM ... [有效分子的原子坐标]
TER
END
```

## 故障排除

### 问题1：没有找到 PDB 文件

```
Found 0 PDB files to process
```

**解决方法**：
- 检查输入目录路径是否正确
- 确保 PDB 文件以 `.pdb` 为扩展名

### 问题2：找不到排除文件

```
Warning: Exclude file ... not found!
```

**解决方法**：
- 检查 exclude_molecules.txt 文件是否存在
- 使用 `--exclude-file` 参数指定正确的路径

### 问题3：内存不足

**解决方法**：
- 减少 `--workers` 参数的值
- 分批处理文件

### 问题4：处理速度慢

**解决方法**：
- 增加 `--workers` 参数（但不要超过 CPU 核心数太多）
- 确保输入/输出目录在快速存储设备上（如 SSD）

## 统计信息说明

脚本会输出以下统计信息：

1. **总体统计**：
   - 处理的文件总数
   - 成功处理的文件数
   - 无有效分子的文件数
   - 错误数

2. **分布统计**：
   - 按有效小分子数量分组的文件数
   - 每组的示例文件和分子类型

3. **详细报告**：
   - 保存在 `ligand_extraction_report.txt`
   - 包含每个文件的详细信息

## 注意事项

1. **文件覆盖**：如果输出目录已存在同名文件，会被覆盖
2. **磁盘空间**：确保有足够的磁盘空间存储输出文件
3. **备份**：建议在运行前备份原始数据
4. **路径**：所有路径都支持相对路径和绝对路径

## 示例输出

```
Loading exclude molecules from processed_ligands/exclude_molecules.txt...
Loaded 67 molecules to exclude: ['1PE', '2MI', '4IP', ...]

Found 5 PDB files to process

Processing with 2 worker processes...

================================================================================
PROCESSING COMPLETE
================================================================================

Total files processed: 5
Successfully processed: 4
Files with no effective molecules: 1
Errors: 0

--------------------------------------------------------------------------------
DISTRIBUTION BY NUMBER OF EFFECTIVE MOLECULES
--------------------------------------------------------------------------------

1 effective molecule(s): 3 files
  Output directory: processed_ligands_effect_1/
    - 100D-assembly1_ligands.pdb: SPM
    - 101M-assembly1_ligands.pdb: NBN
    - 102L-assembly1_ligands.pdb: BME

2 effective molecule(s): 1 files
  Output directory: processed_ligands_effect_2/
    - 101D-assembly1_ligands.pdb: CBR, NT

--------------------------------------------------------------------------------
SUMMARY TABLE
--------------------------------------------------------------------------------
Effective Molecules       Number of Files
--------------------------------------------------------------------------------
1                         3
2                         1
0 (no effective)          1
--------------------------------------------------------------------------------

Detailed report saved to: ./ligand_extraction_report.txt

Done!
```

## 联系方式

如有问题或建议，请联系开发者。
