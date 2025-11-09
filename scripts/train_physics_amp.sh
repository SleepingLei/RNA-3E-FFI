#!/bin/bash

echo "=========================================="
echo "Multi-GPU Training with AMP Optimization"
echo "=========================================="

# Run training with mixed precision and multi-GPU support
# This configuration uses:
# - 4 GPUs (--use_ddp --world_size 4)
# - Automatic Mixed Precision (--use_amp) for ~50% memory reduction
# - Reduced model size (--num_layers 4 instead of 6)
# - Batch size 1 per GPU
# - Gradient accumulation 2 steps

python scripts/04_train_model.py \
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
    --save_every 5 \
    --loss_fn mse \
    --monitor_gradients \
    --num_layers 4 \
    --dropout 0.1 \
    --output_dir models/physics_amp_4gpu \
    --use_ddp \
    --world_size 4 \
    --use_amp

# Alternative: Resume training
# Uncomment the following lines to resume from checkpoint:
# --resume \
# --checkpoint models/physics_amp_4gpu/best_model.pt

echo ""
echo "Training completed!"
