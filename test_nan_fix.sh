#!/bin/bash
# 快速测试NaN修复是否有效

echo "========================================"
echo "测试NaN修复"
echo "========================================"
echo ""
echo "配置: 6层, 无中间LayerNorm, Cosine Loss"
echo "之前: 会立即出现NaN"
echo "修复后: 应该正常训练"
echo ""

python scripts/04_train_model.py \
    --embeddings_path data/processed/ligand_embeddings.h5 \
    --output_dim 1536 \
    --batch_size 4 \
    --num_epochs 3 \
    --lr 5e-4 \
    --num_workers 1 \
    --use_multi_hop \
    --use_nonbonded \
    --use_gate \
    --save_every 10 \
    --num_layers 6 \
    --dropout 0.1 \
    --loss_fn cosine \
    --monitor_gradients \
    --output_dir models/checkpoints_nan_test

echo ""
echo "========================================"
echo "如果看到正常的Loss值（不是NaN），说明修复成功！"
echo "========================================"
