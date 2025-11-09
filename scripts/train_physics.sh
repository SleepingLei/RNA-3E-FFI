#!/bin/bash

echo "=========================================="
echo "Training with Memory Optimization"
echo "=========================================="

# Run training with optimized settings
python scripts/04_train_model.py \
    --embeddings_path data/processed/ligand_embeddings.h5 \
    --output_dim 1536 \
    --batch_size 8 \
    --num_epochs 300 \
    --lr 2e-4 \
    --num_workers 1 \
    --use_multi_hop \
    --use_nonbonded \
    --use_gate \
    --save_every 5 \
    --loss_fn mse --monitor_gradients \
    --num_layers 6 --use_layer_norm --dropout 0.1 \
    --output_dir models/physics_6_0.1 \
    --use_ddp --world_size 4 --splits_file data/splits/filtered_splits.json \
    --resume --checkpoint models/physics_6_0.1/checkpoint_epoch_120.pt
    #--accumulation_step 2 \
echo ""
echo "Training completed!"
