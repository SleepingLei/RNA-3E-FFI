#!/bin/bash
# Optimized training script with memory management

# Set PyTorch memory allocation settings to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Optional: Enable memory debugging (comment out for production)
# export CUDA_LAUNCH_BLOCKING=1

# Reduce batch size if still having OOM issues
BATCH_SIZE=${1:-2}  # Default to 2, can pass as first argument

echo "=========================================="
echo "Training with Memory Optimization"
echo "=========================================="
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "Batch size: $BATCH_SIZE"
echo ""

# Run training with optimized settings
python scripts/04_train_model.py \
    --hariboss_csv hariboss/Complexes.csv \
    --graph_dir data/processed/graphs \
    --embeddings_path data/processed/ligand_embeddings.h5 \
    --splits_file data/splits/splits.json \
    --output_dim 1536 \
    --num_layers 4 \
    --batch_size $BATCH_SIZE \
    --num_epochs 300 \
    --lr 0.001 \
    --num_workers 2 \
    --use_multi_hop \
    --use_nonbonded \
    --use_gate \
    --use_layer_norm \
    --save_every 3

echo ""
echo "Training completed!"
