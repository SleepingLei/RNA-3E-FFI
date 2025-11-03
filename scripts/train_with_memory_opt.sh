#!/bin/bash
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
    --num_layers 4 \
    #--use_layer_norm \
    --dropout 0.1 \
    --loss_fn cosin \
    --monitor_gradients \
    --output_dir models/checkpoints_mse_1536d
echo ""
echo "Training completed!"
