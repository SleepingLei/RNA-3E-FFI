#!/bin/bash

echo "=========================================="
echo "Training with Memory Optimization"
echo "=========================================="

# Run training with optimized settings
python scripts/04_train_model.py \
    --embeddings_path data/processed/ligand_embeddings.h5 \
    --output_dim 1536 \
    --batch_size 1 \
    --num_epochs 300 \
    --lr 0.001 \
    --num_workers 1 \
    --use_multi_hop \
    --use_nonbonded \
    --use_gate \
    --save_every 5 \
    --loss_fn mse --monitor_gradients \
    --num_layers 6 --use_layer_norm --dropout 0.1 \
    --hidden_irreps "64x0e + 32x1o + 16x2e" \
    --output_dir models/checkpoints_mse_1536_6_dropout_0.1_64_32_16 \
    --resume --checkpoint models/checkpoints_mse_1536_6_dropout_0.1_64_32_16/best_model.pt
echo ""
echo "Training completed!"
