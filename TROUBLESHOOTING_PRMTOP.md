# Troubleshooting Empty PRMTOP Files

## 问题描述

在运行 `03_build_dataset.py` 时遇到错误：
```
Could not identify file format
parmed.exceptions.FormatNotFound: Could not identify file format
```

**根本原因**：某些 `.prmtop` 文件是空的（0字节），这是因为 `tleap` 参数化失败但没有生成错误文件。

---

## 诊断步骤

### 1. 运行诊断脚本检查所有 prmtop 文件

```bash
# 检查所有 prmtop 文件的状态
python scripts/debug_prmtop_files.py --amber_dir data/processed/amber

# 检查特定的问题文件
python scripts/debug_prmtop_files.py --amber_dir data/processed/amber --check_specific 7ych 7yci
```

**输出内容**：
- 空文件列表（0字节）
- 异常小的文件（< 1KB）
- 缺失配对文件（.inpcrd 或 .pdb）
- 格式错误的文件

### 2. 分析处理结果

```bash
# 分析 01_process_data.py 的处理结果
python scripts/analyze_failed_parameterization.py \
    --results_file data/processing_results.json \
    --amber_dir data/processed/amber
```

**输出内容**：
- RNA 参数化成功/失败统计
- 空文件的详细信息
- tleap 脚本残留（表示失败）

---

## 常见原因和解决方案

### 原因 1: RNA 片段的末端原子问题

**症状**：
- O5' 和 O3' 原子数量不匹配
- tleap 无法识别末端类型

**解决方案**：
`01_process_data.py` 中的 `clean_rna_terminal_atoms()` 函数应该已经处理了这个问题。检查：

```bash
# 查看清理后的 PDB 文件
ls data/processed/amber/*_rna_cleaned.pdb | head -10
```

如果没有生成 `_rna_cleaned.pdb` 文件，说明清理步骤被跳过了。

### 原因 2: Modified RNA 残基

**症状**：
- 包含非标准核苷酸（PSU, 5MU, 7MG 等）
- tleap 的 RNA.OL3 力场不认识这些残基

**解决方案**：
1. 检查哪些复合物包含 modified RNA：
```python
import json
with open('data/processing_results.json') as f:
    results = json.load(f)
    for r in results:
        if 'modified_rna' in r.get('components', {}):
            print(r['pdb_id'], r['ligand'], r['components']['modified_rna'])
```

2. 这些复合物应该在 `process_complex_v2()` 中单独处理，不应该混在标准 RNA 中。

### 原因 3: 内存或超时问题

**症状**：
- 大型复合物的 prmtop 文件为空
- 没有明显的错误信息

**解决方案**：
检查 tleap 是否因为内存或超时被中断：

```bash
# 查找最大的 RNA 结构
python -c "
import json
with open('data/processing_results.json') as f:
    results = json.load(f)
    rna_sizes = [(r['pdb_id'], r['ligand'], r['model_id'],
                  r['components'].get('rna', {}).get('atoms', 0))
                 for r in results if 'rna' in r.get('components', {})]
    rna_sizes.sort(key=lambda x: x[3], reverse=True)
    print('Top 10 largest RNA structures:')
    for pdb, lig, model, atoms in rna_sizes[:10]:
        print(f'  {pdb}_{lig}_model{model}: {atoms} atoms')
"
```

### 原因 4: tleap 脚本错误

**症状**：
- 残留的 `*_tleap.in` 文件（说明 tleap 失败，cleanup 被跳过）

**解决方案**：
手动检查 tleap 输入脚本：

```bash
# 查找残留的 tleap 脚本
find data/processed/amber -name "*_tleap.in" | head -5

# 查看一个示例
cat data/processed/amber/7ych_GTP_model0_rna_tleap.in
```

手动运行 tleap 查看详细错误：

```bash
cd data/processed/amber
tleap -f 7ych_GTP_model0_rna_tleap.in > tleap_debug.log 2>&1
cat tleap_debug.log
```

---

## 修复步骤

### 步骤 1: 清理失败的文件

```bash
# 删除所有空的 prmtop 文件
find data/processed/amber -name "*.prmtop" -size 0 -delete

# 删除对应的 inpcrd 文件
find data/processed/amber -name "*.inpcrd" -size 0 -delete

# 删除残留的 tleap 脚本
find data/processed/amber -name "*_tleap.in" -delete
```

### 步骤 2: 识别需要重新处理的复合物

```bash
# 生成需要重新处理的 PDB ID 列表
python scripts/debug_prmtop_files.py --amber_dir data/processed/amber > prmtop_check.log

# 从失败列表中提取 PDB IDs
grep "EMPTY\|TOO_SMALL" prmtop_check.log | awk '{print $2}' | \
    sed 's/_.*//g' | sort -u > failed_pdb_ids.txt
```

### 步骤 3: 重新运行处理（针对失败的复合物）

如果失败的数量不多（<100），可以手动重新处理：

```bash
# 创建一个包含失败 PDB IDs 的小 CSV 文件
python -c "
import pandas as pd

# 读取原始 HARIBOSS CSV
df = pd.read_csv('hariboss/Complexes.csv')

# 读取失败的 PDB IDs
with open('failed_pdb_ids.txt') as f:
    failed_ids = set(line.strip() for line in f)

# 过滤
failed_df = df[df['id'].str.lower().isin(failed_ids)]
failed_df.to_csv('hariboss/Failed_Complexes.csv', index=False)

print(f'Found {len(failed_df)} failed complexes to reprocess')
"

# 重新处理
python scripts/01_process_data.py \
    --hariboss_csv hariboss/Failed_Complexes.csv \
    --output_dir data \
    --pocket_cutoff 5.0 \
    --num_workers 4
```

### 步骤 4: 跳过无法修复的复合物

如果某些复合物始终失败，可以创建一个排除列表：

```bash
# 创建排除列表
cat > data/processed/excluded_complexes.txt << EOF
7ych_GTP
7yci_GTP
EOF

# 修改 HARIBOSS CSV 以排除这些复合物（可选）
```

---

## 预防措施

### 在 `scripts/01_process_data.py` 中添加更详细的日志

修改 `parameterize_rna()` 函数，保存 tleap 的输出：

```python
# 在运行 tleap 后添加：
if not (prmtop_file.exists() and inpcrd_file.exists()):
    print(f"  ✗ tleap failed")
    # 保存 tleap 输出用于调试
    log_file = output_prefix.parent / f"{output_prefix.stem}_rna_tleap.log"
    with open(log_file, 'w') as f:
        f.write("=== STDOUT ===\n")
        f.write(result.stdout if result.stdout else "No output\n")
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr if result.stderr else "No errors\n")
    print(f"  Log saved to {log_file.name}")
```

### 在 `scripts/03_build_dataset.py` 中更好的错误处理

已经添加了空文件检测（第265-272行）：

```python
# Check if prmtop file is empty or too small
prmtop_size = rna_prmtop_path.stat().st_size
if prmtop_size == 0:
    failed_list.append((complex_model_id, "rna_prmtop_empty (0 bytes)"))
    continue
elif prmtop_size < 100:  # Suspiciously small
    failed_list.append((complex_model_id, f"rna_prmtop_too_small ({prmtop_size} bytes)"))
    continue
```

---

## 快速检查脚本

保存为 `quick_check.sh`：

```bash
#!/bin/bash

echo "=== Quick PRMTOP Health Check ==="
echo ""

AMBER_DIR="data/processed/amber"

echo "Total prmtop files: $(ls $AMBER_DIR/*.prmtop 2>/dev/null | wc -l)"
echo "Empty prmtop files: $(find $AMBER_DIR -name "*.prmtop" -size 0 2>/dev/null | wc -l)"
echo "Small prmtop files (<1KB): $(find $AMBER_DIR -name "*.prmtop" -size -1k 2>/dev/null | wc -l)"
echo "Orphaned tleap scripts: $(ls $AMBER_DIR/*_tleap.in 2>/dev/null | wc -l)"
echo ""

echo "Sample of empty prmtop files:"
find $AMBER_DIR -name "*.prmtop" -size 0 2>/dev/null | head -5

echo ""
echo "To fix: run the diagnostic scripts mentioned above"
```

---

## 总结

**主要问题**：tleap 参数化失败但生成了空文件

**直接解决方案**：
1. 使用 `debug_prmtop_files.py` 识别所有问题文件
2. 删除空文件和残留脚本
3. 针对失败的复合物重新运行 `01_process_data.py`
4. 对于无法修复的复合物，添加到排除列表

**长期解决方案**：
1. 改进 `01_process_data.py` 的错误日志
2. 在 `03_build_dataset.py` 中添加更好的验证
3. 为 modified RNA 单独处理
4. 考虑使用其他力场或参数化工具

**现在运行**：
```bash
# 在远程服务器上运行
python scripts/debug_prmtop_files.py --amber_dir data/processed/amber
python scripts/analyze_failed_parameterization.py
```
