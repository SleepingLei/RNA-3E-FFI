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
    --output_dim 1536 \
    --batch_size 1 \
    --num_epochs 300 \
    --lr 0.001 \
    --num_workers 1 \
    --use_nonbonded \
    --use_multi_hop \
    --use_gate \
    --save_every 5 \
    --num_layers 4 --use_layer_norm --dropout 0.10 \
    --output_dir models/checkpoints_v2_normalized_1536_4_dropout_0.10 \
    #--resume --checkpoint models/checkpoints_v2_normalized_1536_4_dropout_0.10/best_model.pt
echo ""
echo "Training completed!"
