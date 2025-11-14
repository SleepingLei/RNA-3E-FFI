#!/bin/bash
# 快速运行脚本 - 分类和复制受体文件

# 设置默认参数
LIGAND_DIR="processed_ligands_effect_1"
RECEPTOR_DIR="processed_polymers_fixed"
OUTPUT_DIR="effect_receptor"
WORKERS=$(python3 -c "import multiprocessing; print(multiprocessing.cpu_count())")
ALL_LIGAND_DIRS=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            ALL_LIGAND_DIRS=true
            shift
            ;;
        --ligand-dir)
            LIGAND_DIR="$2"
            shift 2
            ;;
        --receptor-dir)
            RECEPTOR_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--all] [--ligand-dir DIR] [--receptor-dir DIR] [--output-dir DIR] [--workers N]"
            exit 1
            ;;
    esac
done

# 打印配置信息
echo "================================"
echo "受体分类脚本 - 快速运行"
echo "================================"
if [ "$ALL_LIGAND_DIRS" = true ]; then
    echo "配体目录: 所有 processed_ligands_effect_* 目录"
else
    echo "配体目录: $LIGAND_DIR"
fi
echo "受体目录: $RECEPTOR_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "工作进程: $WORKERS"
echo "================================"
echo ""

# 检查必要目录是否存在
if [ "$ALL_LIGAND_DIRS" = false ] && [ ! -d "$LIGAND_DIR" ]; then
    echo "错误：配体目录 $LIGAND_DIR 不存在！"
    exit 1
fi

if [ ! -d "$RECEPTOR_DIR" ]; then
    echo "错误：受体目录 $RECEPTOR_DIR 不存在！"
    exit 1
fi

# 构建命令
CMD="python3 classify_receptors.py --receptor-dir \"$RECEPTOR_DIR\" --output-dir \"$OUTPUT_DIR\" --workers $WORKERS"

if [ "$ALL_LIGAND_DIRS" = true ]; then
    CMD="$CMD --all-ligand-dirs"
else
    CMD="$CMD --ligand-dir \"$LIGAND_DIR\""
fi

# 运行脚本
echo "开始处理..."
eval $CMD

# 检查退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "处理完成！"
    echo "================================"
    echo ""
    echo "生成的文件夹："
    ls -ld $OUTPUT_DIR/*/ 2>/dev/null || echo "没有生成输出文件夹"
    echo ""
    echo "查看详细报告："
    echo "  cat $OUTPUT_DIR/classification_report.txt"
    echo ""
    echo "统计信息："
    if [ -d "$OUTPUT_DIR/RNA" ]; then
        RNA_COUNT=$(ls -1 $OUTPUT_DIR/RNA/*.pdb 2>/dev/null | wc -l)
        echo "  RNA: $RNA_COUNT 个文件"
    fi
    if [ -d "$OUTPUT_DIR/Protein" ]; then
        PROTEIN_COUNT=$(ls -1 $OUTPUT_DIR/Protein/*.pdb 2>/dev/null | wc -l)
        echo "  Protein: $PROTEIN_COUNT 个文件"
    fi
    if [ -d "$OUTPUT_DIR/Complex" ]; then
        COMPLEX_COUNT=$(ls -1 $OUTPUT_DIR/Complex/*.pdb 2>/dev/null | wc -l)
        echo "  Complex: $COMPLEX_COUNT 个文件"
    fi
else
    echo ""
    echo "================================"
    echo "处理失败！请检查错误信息。"
    echo "================================"
    exit 1
fi
