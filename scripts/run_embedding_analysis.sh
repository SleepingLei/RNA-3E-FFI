#!/bin/bash
# Quick Start Script for Embedding Visualization and Analysis
#
# Usage:
#   bash scripts/run_embedding_analysis.sh
#
# Or with custom parameters:
#   bash scripts/run_embedding_analysis.sh \
#       --checkpoint models/checkpoints/custom_model.pt \
#       --graph_dir data/processed/custom_graphs

set -e  # Exit on error

# ============================================================================
# Configuration (modify these paths as needed)
# ============================================================================

# Default paths
CHECKPOINT="${CHECKPOINT:-models/checkpoints_mse_1536_6_dropout_0.1_retry/best_model.pt}"
GRAPH_DIR="${GRAPH_DIR:-data/processed/graphs}"
LIGAND_EMBEDDINGS="${LIGAND_EMBEDDINGS:-data/processed/ligand_embeddings_dedup.h5}"
OUTPUT_DIR="${OUTPUT_DIR:-results/embedding_analysis_retry}"

# Visualization methods (pca tsne umap)
METHODS="${METHODS:-pca tsne}"

# Split selection (empty means all data)
SPLITS_FILE="${SPLITS_FILE:-}"
SPLITS="${SPLITS:-}"

# ============================================================================
# Parse command-line arguments (optional)
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --graph_dir)
            GRAPH_DIR="$2"
            shift 2
            ;;
        --ligand_embeddings)
            LIGAND_EMBEDDINGS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --splits_file)
            SPLITS_FILE="$2"
            shift 2
            ;;
        --splits)
            SPLITS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH          Model checkpoint (default: models/checkpoints/best_model.pt)"
            echo "  --graph_dir PATH           Directory with pocket graphs (default: data/processed/graphs)"
            echo "  --ligand_embeddings PATH   Ligand embeddings HDF5 (default: data/processed/ligand_embeddings_dedup.h5)"
            echo "  --output_dir PATH          Output directory (default: results/embedding_analysis)"
            echo "  --methods 'METHOD1 ...'    Reduction methods (default: 'pca tsne umap')"
            echo "  --splits_file PATH         Path to splits.json file (default: none, uses all data)"
            echo "  --splits 'SPLIT1 ...'      Which splits to use: train, val, test (default: none, uses all data)"
            echo "                             Examples: 'val test', 'train val test', 'test'"
            echo "  --help, -h                 Show this help message"
            echo ""
            echo "Environment variables can also be used:"
            echo "  CHECKPOINT, GRAPH_DIR, LIGAND_EMBEDDINGS, OUTPUT_DIR, METHODS, SPLITS_FILE, SPLITS"
            echo ""
            echo "Examples:"
            echo "  # Analyze all data"
            echo "  $0"
            echo ""
            echo "  # Analyze only test set"
            echo "  $0 --splits_file data/splits/splits.json --splits test"
            echo ""
            echo "  # Analyze validation + test sets"
            echo "  $0 --splits_file data/splits/splits.json --splits 'val test'"
            echo ""
            echo "  # Analyze all splits (train + val + test)"
            echo "  $0 --splits_file data/splits/splits.json --splits 'train val test'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Print Configuration
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║          Pocket-Ligand Embedding Analysis Pipeline                    ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Checkpoint:         $CHECKPOINT"
echo "  Graph Directory:    $GRAPH_DIR"
echo "  Ligand Embeddings:  $LIGAND_EMBEDDINGS"
echo "  Output Directory:   $OUTPUT_DIR"
echo "  Reduction Methods:  $METHODS"
if [ -n "$SPLITS_FILE" ] && [ -n "$SPLITS" ]; then
    echo "  Splits File:        $SPLITS_FILE"
    echo "  Selected Splits:    $SPLITS"
else
    echo "  Splits:             Using all available data"
fi
echo ""

# ============================================================================
# Validation
# ============================================================================

echo "Validating inputs..."

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -d "$GRAPH_DIR" ]; then
    echo "Error: Graph directory not found: $GRAPH_DIR"
    exit 1
fi

if [ ! -f "$LIGAND_EMBEDDINGS" ]; then
    echo "Error: Ligand embeddings not found: $LIGAND_EMBEDDINGS"
    exit 1
fi

# Check number of graph files
N_GRAPHS=$(find "$GRAPH_DIR" -name "*.pt" | wc -l | tr -d ' ')
if [ "$N_GRAPHS" -eq 0 ]; then
    echo "Error: No .pt files found in $GRAPH_DIR"
    exit 1
fi

echo "✓ Found $N_GRAPHS pocket graph files"

# Check if ligand embeddings file is valid
if ! python3 -c "import h5py; h5py.File('$LIGAND_EMBEDDINGS', 'r')" 2>/dev/null; then
    echo "Error: Invalid HDF5 file: $LIGAND_EMBEDDINGS"
    exit 1
fi

N_LIGANDS=$(python3 -c "import h5py; print(len(h5py.File('$LIGAND_EMBEDDINGS', 'r').keys()))")
echo "✓ Found $N_LIGANDS unique ligands in embeddings"

echo "✓ All inputs validated"
echo ""

# ============================================================================
# Step 1: Main Visualization
# ============================================================================

VIZ_DIR="$OUTPUT_DIR/visualizations"

echo "═══════════════════════════════════════════════════════════════════════"
echo "Step 1: Running Main Visualization Analysis"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Build command with optional splits parameters
CMD="python3 scripts/visualize_embeddings.py \
    --checkpoint \"$CHECKPOINT\" \
    --graph_dir \"$GRAPH_DIR\" \
    --ligand_embeddings \"$LIGAND_EMBEDDINGS\" \
    --output_dir \"$VIZ_DIR\" \
    --methods $METHODS"

if [ -n "$SPLITS_FILE" ] && [ -n "$SPLITS" ]; then
    CMD="$CMD --splits_file \"$SPLITS_FILE\" --splits $SPLITS"
fi

eval $CMD

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Visualization step failed"
    exit 1
fi

echo ""
echo "✓ Step 1 completed successfully"
echo ""

# ============================================================================
# Step 2: Advanced Analysis
# ============================================================================

ADVANCED_DIR="$OUTPUT_DIR/advanced_analysis"
MATCHED_PAIRS="$VIZ_DIR/matched_pairs.json"

echo "═══════════════════════════════════════════════════════════════════════"
echo "Step 2: Running Advanced Analysis"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

if [ ! -f "$MATCHED_PAIRS" ]; then
    echo "Error: matched_pairs.json not found: $MATCHED_PAIRS"
    echo "Step 1 may have failed to generate this file"
    exit 1
fi

python3 scripts/advanced_embedding_analysis.py \
    --matched_pairs "$MATCHED_PAIRS" \
    --output_dir "$ADVANCED_DIR"

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Advanced analysis step failed"
    exit 1
fi

echo ""
echo "✓ Step 2 completed successfully"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                    Analysis Pipeline Completed!                       ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "📁 Output Structure:"
echo "   $OUTPUT_DIR/"
echo "   ├── visualizations/"
echo "   │   ├── Data Files:"
echo "   │   │   ├── pocket_embeddings.npz"
echo "   │   │   ├── matched_pairs.json"
echo "   │   │   ├── pocket_ligand_distances.csv"
echo "   │   │   ├── pocket_ligand_correlations.csv"
echo "   │   │   ├── ligand_summary.csv"
echo "   │   │   └── ligand_distance_stats.csv"
echo "   │   ├── Visualizations:"
echo "   │   │   ├── joint_pca_*.png/pdf (3 files)"
echo "   │   │   ├── joint_tsne_*.png/pdf (3 files)"
echo "   │   │   ├── joint_umap_*.png/pdf (3 files, if UMAP available)"
echo "   │   │   ├── distance_distributions.png/pdf"
echo "   │   │   ├── correlation_distributions.png/pdf"
echo "   │   │   └── ligand_distribution.png/pdf"
echo "   │   └── analysis_report.md"
echo "   └── advanced_analysis/"
echo "       ├── clustering_optimization.png"
echo "       ├── kmeans_clusters.png"
echo "       ├── cluster_assignments.csv"
echo "       ├── retrieval_performance.png"
echo "       ├── retrieval_results.csv"
echo "       ├── intra_inter_distances.png"
echo "       ├── intra_inter_boxplot.png"
echo "       ├── intra_inter_distances.csv"
echo "       ├── ligand_similarity_heatmap.png"
echo "       ├── ligand_distance_matrix.csv"
echo "       └── ligand_dendrogram.png"
echo ""

# Count output files
N_VIZ_FILES=$(find "$VIZ_DIR" -type f | wc -l | tr -d ' ')
N_ADV_FILES=$(find "$ADVANCED_DIR" -type f | wc -l | tr -d ' ')
TOTAL_FILES=$((N_VIZ_FILES + N_ADV_FILES))

echo "📊 Statistics:"
echo "   Total output files: $TOTAL_FILES"
echo "   - Visualization files: $N_VIZ_FILES"
echo "   - Advanced analysis files: $N_ADV_FILES"
echo ""

# Extract key metrics from report if available
REPORT_FILE="$VIZ_DIR/analysis_report.md"
if [ -f "$REPORT_FILE" ]; then
    echo "📈 Key Metrics (from analysis_report.md):"
    echo ""
    grep -A 4 "## Distance Metrics" "$REPORT_FILE" | tail -5 || true
    echo ""
fi

echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Next Steps:"
echo "  1. View the main report:     cat $VIZ_DIR/analysis_report.md"
echo "  2. Explore visualizations:   open $VIZ_DIR/*.png"
echo "  3. Analyze CSV data:         open $VIZ_DIR/*.csv"
echo "  4. Check retrieval results:  open $ADVANCED_DIR/retrieval_results.csv"
echo ""
echo "For detailed documentation, see: scripts/README_embedding_visualization.md"
echo ""
