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
    --weight_decay 2e-6 \
    --num_workers 1 \
    --use_multi_hop \
    --use_nonbonded \
    --use_gate \
    --save_every 5 \
    --loss_fn mse --monitor_gradients \
    --num_layers 6 --use_layer_norm --dropout 0.1 \
    --output_dir models/physics_v3_6_0.1_new_improve_multihead_improved_layers_and_enhanced_invariants \
    --use_ddp --world_size 4 --splits_file data/splits/filtered_splits.json \
    --use_v3_model --num_attention_heads 8 --grad_clip 2.0 --initial_angle_weight 0.33 --initial_dihedral_weight 0.33 --initial_nonbonded_weight 0.33 \
    --use_enhanced_invariants \
    --use_improved_layers \
    #--use_physics_loss \
    #--resume --checkpoint models/physics_v3_6_0.1/checkpoint_epoch_120.pt
    #--accumulation_step 2 \
echo ""
echo "Training completed!"
