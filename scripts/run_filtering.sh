#!/bin/bash

echo "=========================================="
echo "Pre-filtering Large Samples"
echo "=========================================="

# Option 1: Auto-filter (recommended - removes top 5%)
echo ""
echo "Running automatic filtering (removes top 5% largest samples)..."
python scripts/filter_large_samples.py \
    --graph_dir data/processed/graphs \
    --splits_file data/splits/splits.json \
    --output_splits data/splits/filtered_splits.json \
    --percentile 95 \
    --output_dir data/analysis

# Option 2: Aggressive filtering (removes top 1%)
# Uncomment if you still get OOM after Option 1
# python scripts/filter_large_samples.py \
#     --percentile 99 \
#     --output_splits data/splits/filtered_splits_99.json

# Option 3: Custom thresholds
# Uncomment and adjust thresholds as needed
# python scripts/filter_large_samples.py \
#     --max_nodes 1000 \
#     --max_edges 5000 \
#     --max_memory_mb 100 \
#     --output_splits data/splits/filtered_splits_custom.json

echo ""
echo "=========================================="
echo "Filtering completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Check the analysis results in data/analysis/"
echo "2. Run training with filtered dataset:"
echo "   bash scripts/train_physics_amp.sh"
echo ""
echo "Or manually with:"
echo "   python scripts/04_train_model.py \\"
echo "       --splits_file data/splits/filtered_splits.json \\"
echo "       --use_amp --use_ddp --world_size 4"
echo ""
