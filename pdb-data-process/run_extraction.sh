#!/bin/bash
# 快速运行脚本 - 提取有效配体小分子

# 设置默认参数
INPUT_DIR="processed_ligands"
EXCLUDE_FILE="processed_ligands/exclude_molecules.txt"
OUTPUT_BASE="./extracted_ligands"
WORKERS=$(python3 -c "import multiprocessing; print(multiprocessing.cpu_count() -10)")

# 打印配置信息
echo "================================"
echo "配体提取脚本 - 快速运行"
echo "================================"
echo "输入目录: $INPUT_DIR"
echo "排除文件: $EXCLUDE_FILE"
echo "输出目录: $OUTPUT_BASE"
echo "工作进程: $WORKERS"
echo "================================"
echo ""

# 检查必要文件是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误：输入目录 $INPUT_DIR 不存在！"
    exit 1
fi

if [ ! -f "$EXCLUDE_FILE" ]; then
    echo "警告：排除文件 $EXCLUDE_FILE 不存在！将不排除任何分子。"
fi

# 运行脚本
echo "开始处理..."
python3 extract_effective_ligands.py \
    --input-dir "$INPUT_DIR" \
    --exclude-file "$EXCLUDE_FILE" \
    --output-base "$OUTPUT_BASE" \
    --workers "$WORKERS"

# 检查退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "处理完成！"
    echo "================================"
    echo ""
    echo "生成的文件夹："
    ls -ld processed_ligands_effect_*/ 2>/dev/null || echo "没有生成输出文件夹"
    echo ""
    echo "查看详细报告："
    echo "  cat ligand_extraction_report.txt"
else
    echo ""
    echo "================================"
    echo "处理失败！请检查错误信息。"
    echo "================================"
    exit 1
fi
