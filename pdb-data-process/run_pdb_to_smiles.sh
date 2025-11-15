#!/bin/bash
# 快速运行脚本 - PDB 转 SMILES

INPUT_DIR="${1:-processed_ligands_effect_1}"
OUTPUT_CSV="${2:-ligands_smiles.csv}"
PH="${3:-7.4}"
WORKERS=$(python3 -c "import multiprocessing; print(multiprocessing.cpu_count() - 10)")

echo "================================"
echo "PDB 转 SMILES"
echo "================================"
echo "输入目录: $INPUT_DIR"
echo "输出文件: $OUTPUT_CSV"
echo "pH 值: $PH"
echo "工作进程: $WORKERS"
echo "================================"
echo ""

# 运行脚本
python3 pdb_to_smiles.py \
    --input-dir "$INPUT_DIR" \
    --output-csv "$OUTPUT_CSV" \
    --ph "$PH" \
    --workers "$WORKERS"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "转换完成！"
    echo "================================"
    echo ""
    echo "查看结果："
    echo "  head -10 $OUTPUT_CSV"
    echo "  cat $OUTPUT_CSV"
else
    echo ""
    echo "================================"
    echo "转换失败！请检查错误信息。"
    echo "================================"
    exit 1
fi
