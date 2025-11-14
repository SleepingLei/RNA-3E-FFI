# PDB 数据处理完整工作流程

本文档描述完整的 PDB 数据处理流程，包括配体提取和受体分类。

## 概述

本工作流程包含两个主要步骤：

1. **配体提取**：从 PDB 配体文件中提取有效的小分子，排除常见的辅因子和溶剂分子
2. **受体分类**：根据受体组成（RNA、蛋白质或复合物）对受体文件进行分类

## 目录结构

```
pdb-data-process/
├── processed_ligands/              # 原始配体文件
│   ├── *_ligands.pdb
│   └── exclude_molecules.txt       # 需要排除的分子列表
├── processed_polymers_fixed/       # 原始受体文件
│   └── *_polymer_fixed.pdb
├── processed_ligands_effect_1/     # 提取结果：1个有效小分子
├── processed_ligands_effect_2/     # 提取结果：2个有效小分子
├── processed_ligands_effect_N/     # 提取结果：N个有效小分子
└── effect_receptor/                # 受体分类结果
    ├── RNA/                        # RNA 受体
    ├── Protein/                    # 蛋白质受体
    ├── Complex/                    # RNA-蛋白质复合物
    └── classification_report.txt   # 分类报告
```

## 完整工作流程

### 步骤 0：准备数据

确保你有以下数据：

1. `processed_ligands/` 目录，包含所有配体 PDB 文件
2. `processed_polymers_fixed/` 目录，包含所有受体 PDB 文件
3. `processed_ligands/exclude_molecules.txt` 文件，包含需要排除的分子列表

### 步骤 1：提取有效配体

```bash
# 方法1：使用快速脚本（推荐）
./run_extraction.sh

# 方法2：使用 Python 脚本
python extract_effective_ligands.py --workers 16
```

**输出**：
- 生成 `processed_ligands_effect_N/` 目录（N 为有效小分子数量）
- 生成 `ligand_extraction_report.txt` 报告

**预期结果示例**：
```
processed_ligands_effect_1/  # 含1个有效小分子：3个文件
processed_ligands_effect_2/  # 含2个有效小分子：1个文件
```

### 步骤 2：分类受体

```bash
# 方法1：使用快速脚本处理所有配体目录（推荐）
./run_classification.sh --all

# 方法2：使用 Python 脚本
python classify_receptors.py --all-ligand-dirs --workers 16

# 方法3：只处理特定配体目录
./run_classification.sh --ligand-dir processed_ligands_effect_1
```

**输出**：
- 生成 `effect_receptor/RNA/` 目录（RNA 受体）
- 生成 `effect_receptor/Protein/` 目录（蛋白质受体）
- 生成 `effect_receptor/Complex/` 目录（复合物受体，如果有）
- 生成 `effect_receptor/classification_report.txt` 报告

**预期结果示例**：
```
effect_receptor/
├── RNA/                          # 2个 RNA 受体
│   ├── 100D-assembly1_polymer_fixed.pdb
│   └── 101D-assembly1_polymer_fixed.pdb
├── Protein/                      # 2个蛋白质受体
│   ├── 101M-assembly1_polymer_fixed.pdb
│   └── 102L-assembly1_polymer_fixed.pdb
└── classification_report.txt
```

### 步骤 3：查看结果

```bash
# 查看配体提取报告
cat ligand_extraction_report.txt

# 查看受体分类报告
cat effect_receptor/classification_report.txt

# 统计各类型数量
echo "RNA 受体: $(ls -1 effect_receptor/RNA/*.pdb 2>/dev/null | wc -l)"
echo "蛋白质受体: $(ls -1 effect_receptor/Protein/*.pdb 2>/dev/null | wc -l)"
echo "复合物受体: $(ls -1 effect_receptor/Complex/*.pdb 2>/dev/null | wc -l)"
```

## 本地测试结果

### 配体提取结果

| 有效小分子数 | 文件数 | 示例 |
|------------|-------|-----|
| 1个 | 3 | SPM, NBN, BME |
| 2个 | 1 | CBR + NT |
| 0个（仅排除分子） | 1 | - |

**总计**：5 个配体文件，4 个包含有效小分子

### 受体分类结果

| 分类 | 数量 | 百分比 | 示例残基 |
|-----|------|--------|---------|
| RNA | 2 | 50.00% | C, DC, DG, G / DA, DC, DG, DT |
| Protein | 2 | 50.00% | ALA, ARG, ASN, ASP, ... |
| Complex | 0 | 0% | - |

**总计**：4 个受体文件，2 个 RNA，2 个蛋白质

## 在远程机器上运行

### 准备工作

```bash
# 1. 上传脚本
scp extract_effective_ligands.py user@remote:/path/to/dir/
scp classify_receptors.py user@remote:/path/to/dir/
scp run_extraction.sh user@remote:/path/to/dir/
scp run_classification.sh user@remote:/path/to/dir/

# 2. 上传数据（如果需要）
rsync -avz processed_ligands/ user@remote:/path/to/dir/processed_ligands/
rsync -avz processed_polymers_fixed/ user@remote:/path/to/dir/processed_polymers_fixed/
```

### 执行处理

```bash
# 登录到远程机器
ssh user@remote
cd /path/to/dir/

# 方法1：使用 screen（推荐）
screen -S pdb_processing

# 步骤1：提取配体
./run_extraction.sh
# 等待完成...

# 步骤2：分类受体
./run_classification.sh --all
# 等待完成...

# 分离 screen 会话：Ctrl+A+D

# 方法2：使用 nohup
nohup bash -c "./run_extraction.sh && ./run_classification.sh --all" > processing.log 2>&1 &
```

### 监控进度

```bash
# 如果使用 screen
screen -r pdb_processing

# 如果使用 nohup
tail -f processing.log

# 查看当前进度
ls -la processed_ligands_effect_*/
ls -la effect_receptor/
```

### 下载结果

```bash
# 下载分类后的受体文件
rsync -avz user@remote:/path/to/dir/effect_receptor/ ./effect_receptor/

# 下载报告
scp user@remote:/path/to/dir/ligand_extraction_report.txt ./
scp user@remote:/path/to/dir/effect_receptor/classification_report.txt ./
```

## 配置参数说明

### 工作进程数（--workers）

根据机器配置选择合适的工作进程数：

| 机器类型 | CPU 核心数 | 推荐 workers |
|---------|-----------|-------------|
| 个人电脑 | 4-8 | 4-8 |
| 工作站 | 16-32 | 16-32 |
| 服务器 | 64+ | 32-64 |

**经验法则**：
- CPU 密集型：workers = CPU 核心数
- I/O 密集型：workers = CPU 核心数 × 1.5-2

### 内存使用

- 每个 worker 进程大约需要 100-500 MB 内存
- 建议预留总内存的 20-30% 给系统

**示例计算**：
```
可用内存: 16 GB
建议使用: 12 GB (16 × 0.75)
每进程内存: 300 MB
最大 workers: 12000 / 300 = 40
实际推荐: min(CPU核心数, 40)
```

## 故障排除

### 问题1：内存不足

**症状**：进程被 killed，或系统响应缓慢

**解决方法**：
```bash
# 减少 workers 数量
./run_extraction.sh --workers 4
./run_classification.sh --all --workers 4
```

### 问题2：磁盘空间不足

**症状**：无法创建输出文件

**解决方法**：
```bash
# 检查磁盘空间
df -h

# 清理临时文件
rm -rf /tmp/*

# 使用其他磁盘
python extract_effective_ligands.py --output-base /other/disk/path
```

### 问题3：找不到受体文件

**症状**：大量 "Receptor not found" 警告

**解决方法**：
1. 检查文件命名是否匹配
2. 确认 `processed_polymers_fixed/` 目录路径正确
3. 验证文件名格式：`<PDB_ID>_polymer_fixed.pdb`

### 问题4：进程速度慢

**解决方法**：
1. 增加 workers 数量（如果 CPU 未满载）
2. 使用 SSD 存储
3. 检查是否有其他进程占用资源（`top` 或 `htop`）

## 批处理技巧

### 大规模数据处理

对于数千个文件的大规模处理：

```bash
# 1. 分批处理配体提取（按字母分组）
for prefix in {A..Z}; do
    mkdir -p temp_ligands_$prefix
    mv processed_ligands/${prefix}*_ligands.pdb temp_ligands_$prefix/
    python extract_effective_ligands.py \
        --input-dir temp_ligands_$prefix \
        --output-base output_$prefix \
        --workers 16
done

# 2. 合并结果
for i in {1..10}; do
    mkdir -p processed_ligands_effect_$i
    cat output_*/processed_ligands_effect_$i/* > processed_ligands_effect_$i/
done

# 3. 分类受体
python classify_receptors.py --all-ligand-dirs --workers 32
```

### 并行处理多个目录

```bash
# 同时处理多个配体目录
python classify_receptors.py --ligand-dir processed_ligands_effect_1 --output-dir output_1 &
python classify_receptors.py --ligand-dir processed_ligands_effect_2 --output-dir output_2 &
python classify_receptors.py --ligand-dir processed_ligands_effect_3 --output-dir output_3 &
wait
```

## 质量检查

### 检查配体提取质量

```bash
# 检查是否有空文件
find processed_ligands_effect_* -name "*.pdb" -size 0

# 检查文件完整性（应该以 END 结尾）
for f in processed_ligands_effect_*/*.pdb; do
    if ! tail -1 "$f" | grep -q "END"; then
        echo "警告：$f 可能不完整"
    fi
done

# 统计提取率
total=$(ls processed_ligands/*.pdb | wc -l)
effective=$(cat ligand_extraction_report.txt | grep "Successfully processed" | awk '{print $3}')
echo "提取成功率：$(echo "scale=2; $effective/$total*100" | bc)%"
```

### 检查受体分类质量

```bash
# 验证文件对应关系
for f in effect_receptor/RNA/*.pdb; do
    pdb_id=$(basename $f | sed 's/_polymer_fixed.pdb//')
    if ! ls processed_ligands_effect_*/${pdb_id}_ligands.pdb 2>/dev/null; then
        echo "警告：$pdb_id 的配体文件未找到"
    fi
done

# 检查分类完整性
ligand_count=$(ls processed_ligands_effect_*/*.pdb | wc -l)
receptor_count=$(find effect_receptor -name "*.pdb" -not -path "*/classification_report.txt" | wc -l)
echo "配体文件数：$ligand_count"
echo "受体文件数：$receptor_count"
```

## 性能基准

基于示例数据的性能参考：

| 任务 | 文件数 | Workers | 时间 | 速度 |
|-----|-------|---------|------|------|
| 配体提取 | 5 | 2 | ~2秒 | 2.5 文件/秒 |
| 受体分类 | 4 | 2 | ~1秒 | 4 文件/秒 |

**预估**（基于线性扩展）：
- 1000 个文件，16 workers：约 1-2 分钟
- 10000 个文件，32 workers：约 10-15 分钟

## 数据备份建议

```bash
# 处理前备份原始数据
tar -czf backup_$(date +%Y%m%d).tar.gz \
    processed_ligands/ \
    processed_polymers_fixed/

# 处理后备份结果
tar -czf results_$(date +%Y%m%d).tar.gz \
    processed_ligands_effect_*/ \
    effect_receptor/ \
    *_report.txt
```

## 脚本列表

| 脚本 | 用途 | 推荐使用 |
|-----|------|---------|
| `extract_effective_ligands.py` | 提取有效配体 | ✓ |
| `classify_receptors.py` | 分类受体 | ✓ |
| `run_extraction.sh` | 快速运行配体提取 | ✓ 推荐 |
| `run_classification.sh` | 快速运行受体分类 | ✓ 推荐 |

## 文档列表

| 文档 | 内容 |
|-----|------|
| `README_extract_ligands.md` | 配体提取脚本详细说明 |
| `README_classify_receptors.md` | 受体分类脚本详细说明 |
| `README_workflow.md` | 完整工作流程（本文档） |

## 快速参考

```bash
# 完整流程（一行命令）
./run_extraction.sh && ./run_classification.sh --all

# 查看所有统计
cat ligand_extraction_report.txt && \
cat effect_receptor/classification_report.txt

# 快速统计
echo "配体目录数：$(ls -d processed_ligands_effect_* | wc -l)"
echo "RNA 受体：$(ls effect_receptor/RNA/*.pdb 2>/dev/null | wc -l)"
echo "蛋白质受体：$(ls effect_receptor/Protein/*.pdb 2>/dev/null | wc -l)"
echo "复合物受体：$(ls effect_receptor/Complex/*.pdb 2>/dev/null | wc -l)"
```

## 联系方式

如有问题或建议，请联系开发者。
