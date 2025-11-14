# 快速开始指南

## 一分钟快速开始

```bash
# 1. 提取有效配体（从 processed_ligands/ 中提取，排除无效分子）
./run_extraction.sh

# 2. 分类受体（将对应的受体文件复制到 effect_receptor/ 并分类为 RNA/Protein/Complex）
./run_classification.sh --all

# 3. 查看结果
ls -la processed_ligands_effect_*/
ls -la effect_receptor/*/
```

## 文件说明

### 核心脚本（可直接运行）

| 文件 | 功能 | 命令 |
|-----|------|------|
| `run_extraction.sh` | 提取有效配体 | `./run_extraction.sh` |
| `run_classification.sh` | 分类受体 | `./run_classification.sh --all` |

### Python 脚本（支持更多参数）

| 文件 | 功能 |
|-----|------|
| `extract_effective_ligands.py` | 配体提取（Python 实现） |
| `classify_receptors.py` | 受体分类（Python 实现） |

### 文档

| 文件 | 内容 |
|-----|------|
| `README_extract_ligands.md` | 配体提取详细说明 |
| `README_classify_receptors.md` | 受体分类详细说明 |
| `README_workflow.md` | 完整工作流程 |
| `QUICKSTART.md` | 本文档 |

## 工作流程图

```
processed_ligands/                     # 输入：原始配体文件
    ├── XXX_ligands.pdb
    └── exclude_molecules.txt          # 排除列表
         ↓
    [run_extraction.sh]                # 步骤1：提取有效配体
         ↓
processed_ligands_effect_N/            # 输出：按有效分子数分类
    ├── processed_ligands_effect_1/    # 1个有效分子
    ├── processed_ligands_effect_2/    # 2个有效分子
    └── ...
         +
processed_polymers_fixed/              # 输入：原始受体文件
    └── XXX_polymer_fixed.pdb
         ↓
    [run_classification.sh --all]      # 步骤2：分类受体
         ↓
effect_receptor/                       # 输出：按组成分类
    ├── RNA/                           # RNA 受体
    ├── Protein/                       # 蛋白质受体
    ├── Complex/                       # RNA-蛋白质复合物
    └── classification_report.txt      # 分类报告
```

## 常用命令

### 基本用法

```bash
# 提取配体（使用所有 CPU 核心）
./run_extraction.sh

# 分类受体（处理所有 processed_ligands_effect_* 目录）
./run_classification.sh --all

# 查看报告
cat ligand_extraction_report.txt
cat effect_receptor/classification_report.txt
```

### 自定义参数

```bash
# 使用 8 个工作进程
./run_extraction.sh --workers 8

# 只处理 processed_ligands_effect_1 目录
./run_classification.sh --ligand-dir processed_ligands_effect_1

# 指定输出目录
./run_classification.sh --all --output-dir my_output
```

### Python 脚本（更多控制）

```bash
# 提取配体（自定义所有参数）
python extract_effective_ligands.py \
    --input-dir processed_ligands \
    --exclude-file processed_ligands/exclude_molecules.txt \
    --output-base . \
    --workers 16

# 分类受体（处理所有目录）
python classify_receptors.py \
    --all-ligand-dirs \
    --receptor-dir processed_polymers_fixed \
    --output-dir effect_receptor \
    --workers 16

# 分类受体（只处理特定目录）
python classify_receptors.py \
    --ligand-dir processed_ligands_effect_1 \
    --receptor-dir processed_polymers_fixed \
    --output-dir effect_receptor_1
```

## 查看结果

```bash
# 统计配体提取结果
echo "=== 配体提取统计 ==="
for dir in processed_ligands_effect_*/; do
    count=$(ls -1 "$dir"*.pdb 2>/dev/null | wc -l)
    echo "$(basename $dir): $count 个文件"
done

# 统计受体分类结果
echo ""
echo "=== 受体分类统计 ==="
echo "RNA 受体: $(ls -1 effect_receptor/RNA/*.pdb 2>/dev/null | wc -l)"
echo "蛋白质受体: $(ls -1 effect_receptor/Protein/*.pdb 2>/dev/null | wc -l)"
echo "复合物受体: $(ls -1 effect_receptor/Complex/*.pdb 2>/dev/null | wc -l)"
```

## 远程机器运行（三步走）

```bash
# 1. 上传文件
scp *.py *.sh user@remote:/path/to/dir/

# 2. 登录并运行
ssh user@remote
cd /path/to/dir/
nohup bash -c "./run_extraction.sh && ./run_classification.sh --all" > processing.log 2>&1 &

# 3. 监控和下载
tail -f processing.log
rsync -avz user@remote:/path/to/dir/effect_receptor/ ./effect_receptor/
```

## 故障排除（一行命令）

```bash
# 检查 Python 版本
python3 --version

# 检查必要的目录
ls -ld processed_ligands/ processed_polymers_fixed/

# 检查 CPU 核心数
python3 -c "import multiprocessing; print(f'CPU 核心数: {multiprocessing.cpu_count()}')"

# 检查磁盘空间
df -h .

# 检查内存
free -h  # Linux
vm_stat  # macOS

# 测试脚本（处理 1 个文件测试）
python3 extract_effective_ligands.py --workers 1
```

## 本地测试结果

运行示例数据的结果：

```
=== 配体提取 ===
- 输入文件: 5 个
- 成功处理: 4 个
- 1个有效分子: 3 个文件 (SPM, NBN, BME)
- 2个有效分子: 1 个文件 (CBR, NT)

=== 受体分类 ===
- 输入文件: 4 个
- RNA 受体: 2 个 (50%)
- 蛋白质受体: 2 个 (50%)
- 复合物受体: 0 个 (0%)
```

## 性能建议

| 文件数量 | 推荐 Workers | 预计时间 |
|---------|-------------|----------|
| < 100 | 4-8 | < 1分钟 |
| 100-1000 | 8-16 | 1-5分钟 |
| 1000-10000 | 16-32 | 5-30分钟 |
| > 10000 | 32-64 | 30分钟+ |

## 下一步

- 查看详细文档：`README_workflow.md`
- 配体提取详情：`README_extract_ligands.md`
- 受体分类详情：`README_classify_receptors.md`

## 常见问题

**Q: 如何只提取特定类型的配体？**
A: 编辑 `processed_ligands/exclude_molecules.txt`，添加或删除需要排除的分子。

**Q: 如何处理更多配体目录？**
A: 使用 `./run_classification.sh --all` 会自动处理所有 `processed_ligands_effect_*` 目录。

**Q: 如何添加新的 RNA 或蛋白质残基类型？**
A: 编辑 `classify_receptors.py` 中的 `RNA_RESIDUES` 或 `PROTEIN_RESIDUES` 集合。

**Q: 脚本运行速度慢怎么办？**
A: 增加 `--workers` 参数，例如 `--workers 16`。

**Q: 如何验证结果正确性？**
A: 随机抽查几个文件，确认：
- 配体文件只包含有效分子（不在 exclude_molecules.txt 中）
- 受体文件分类正确（RNA/Protein/Complex）

## 完整示例

```bash
#!/bin/bash
# 完整处理流程示例

# 设置参数
WORKERS=16

# 步骤1：提取配体
echo "步骤1：提取有效配体..."
./run_extraction.sh --workers $WORKERS
if [ $? -ne 0 ]; then
    echo "错误：配体提取失败"
    exit 1
fi

# 步骤2：分类受体
echo "步骤2：分类受体..."
./run_classification.sh --all --workers $WORKERS
if [ $? -ne 0 ]; then
    echo "错误：受体分类失败"
    exit 1
fi

# 步骤3：生成统计
echo ""
echo "=== 处理完成 ==="
echo ""
echo "配体提取结果："
for dir in processed_ligands_effect_*/; do
    count=$(ls -1 "$dir"*.pdb 2>/dev/null | wc -l)
    echo "  $(basename $dir): $count 个文件"
done

echo ""
echo "受体分类结果："
echo "  RNA: $(ls -1 effect_receptor/RNA/*.pdb 2>/dev/null | wc -l) 个文件"
echo "  Protein: $(ls -1 effect_receptor/Protein/*.pdb 2>/dev/null | wc -l) 个文件"
echo "  Complex: $(ls -1 effect_receptor/Complex/*.pdb 2>/dev/null | wc -l) 个文件"

echo ""
echo "详细报告："
echo "  - ligand_extraction_report.txt"
echo "  - effect_receptor/classification_report.txt"
```

## 联系方式

如有问题或建议，请联系开发者。
