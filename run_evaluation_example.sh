#!/bin/bash
# Example script for running test set evaluation

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Set Evaluation - Example${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Configuration
CHECKPOINT="models/checkpoints/best_model.pt"
SPLITS="data/splits/splits.json"
GRAPH_DIR="data/processed/graphs"
LIGAND_EMBEDDINGS="data/processed/ligand_embeddings.h5"
OUTPUT_DIR="results/evaluation"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}Step 1: Basic Evaluation (default thresholds: 5%, 10%, 20%)${NC}"
python scripts/evaluate_test_set.py \
    --checkpoint "$CHECKPOINT" \
    --splits "$SPLITS" \
    --graph_dir "$GRAPH_DIR" \
    --ligand_embeddings "$LIGAND_EMBEDDINGS" \
    --output "$OUTPUT_DIR/test_results.json" \
    --metric cosine

echo ""
echo -e "${GREEN}Step 2: Evaluation with custom thresholds${NC}"
python scripts/evaluate_test_set.py \
    --checkpoint "$CHECKPOINT" \
    --splits "$SPLITS" \
    --graph_dir "$GRAPH_DIR" \
    --ligand_embeddings "$LIGAND_EMBEDDINGS" \
    --output "$OUTPUT_DIR/test_results_detailed.json" \
    --metric cosine \
    --top_percentages 1 5 10 15 20 25 30

echo ""
echo -e "${GREEN}Step 3: Evaluation with Euclidean distance${NC}"
python scripts/evaluate_test_set.py \
    --checkpoint "$CHECKPOINT" \
    --splits "$SPLITS" \
    --graph_dir "$GRAPH_DIR" \
    --ligand_embeddings "$LIGAND_EMBEDDINGS" \
    --output "$OUTPUT_DIR/test_results_euclidean.json" \
    --metric euclidean

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Evaluation Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results saved to:"
echo "  - $OUTPUT_DIR/test_results.json"
echo "  - $OUTPUT_DIR/test_results_detailed.json"
echo "  - $OUTPUT_DIR/test_results_euclidean.json"
echo ""
echo "You can now analyze the results using the JSON files."
