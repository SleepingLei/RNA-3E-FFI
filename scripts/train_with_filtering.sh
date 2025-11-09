#!/bin/bash

echo "=========================================="
echo "Complete Training Pipeline with Filtering"
echo "=========================================="

# Step 1: Filter large samples (if not already done)
if [ ! -f "data/splits/filtered_splits.json" ]; then
    echo ""
    echo "[Step 1/2] Filtering large samples..."
    echo "This will remove the largest 5% of samples to prevent OOM"
    echo ""

    python scripts/filter_large_samples.py \
        --graph_dir data/processed/graphs \
        --splits_file data/splits/splits.json \
        --output_splits data/splits/filtered_splits.json \
        --percentile 95 \
        --output_dir data/analysis

    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Filtering failed! Please check the error above."
        exit 1
    fi
else
    echo ""
    echo "[Step 1/2] Using existing filtered splits..."
    echo "File: data/splits/filtered_splits.json"
    echo ""
fi

# Step 2: Train with filtered data + AMP + Multi-GPU
echo ""
echo "[Step 2/2] Starting training with optimized settings..."
echo ""
echo "Configuration:"
echo "  - Filtered dataset (removed top 5% largest samples)"
echo "  - Mixed Precision Training (AMP) for 50% memory reduction"
echo "  - Multi-GPU training (4 GPUs)"
echo "  - 4 layers (reduced from 6 for memory)"
echo "  - Batch size: 1 per GPU"
echo "  - Gradient accumulation: 2 steps"
echo "  - Effective batch size: 8 (1 × 2 × 4)"
echo ""

python scripts/04_train_model.py \
    --splits_file data/splits/filtered_splits.json \
    --embeddings_path data/processed/ligand_embeddings.h5 \
    --output_dim 1536 \
    --batch_size 1 \
    --accumulation_steps 2 \
    --num_epochs 300 \
    --lr 0.001 \
    --num_workers 1 \
    --use_multi_hop \
    --use_nonbonded \
    --use_gate \
    --use_layer_norm \
    --num_layers 4 \
    --dropout 0.1 \
    --save_every 5 \
    --loss_fn mse \
    --monitor_gradients \
    --use_ddp \
    --world_size 4 \
    --use_amp \
    --output_dir models/physics_filtered_amp

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Training completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved to: models/physics_filtered_amp/"
    echo ""
    echo "Analysis files:"
    echo "  - data/analysis/graph_statistics.csv"
    echo "  - data/analysis/removed_samples.csv"
    echo "  - data/analysis/graph_size_distribution.png"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Training failed!"
    echo "=========================================="
    echo ""
    echo "Troubleshooting steps:"
    echo "1. Check if you still get OOM errors"
    echo "2. Try more aggressive filtering:"
    echo "   python scripts/filter_large_samples.py --percentile 90"
    echo "3. Reduce model size further:"
    echo "   --num_layers 3"
    echo "4. Check GPU memory:"
    echo "   nvidia-smi"
    echo ""
    exit 1
fi
