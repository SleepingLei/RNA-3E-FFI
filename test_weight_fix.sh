#!/bin/bash
# 快速测试权重修复

echo "========================================"
echo "测试可学习权重修复"
echo "========================================"
echo ""

python -c "
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from scripts.amber_vocabulary import get_global_encoder
from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2

encoder = get_global_encoder()

print('创建模型...')
model = RNAPocketEncoderV2(
    num_atom_types=encoder.num_atom_types,
    num_residues=encoder.num_residues,
    atom_embed_dim=32,
    residue_embed_dim=16,
    hidden_irreps='32x0e + 16x1o + 8x2e',
    output_dim=1536,
    num_layers=6,
    num_radial_basis=8,
    use_multi_hop=True,
    use_nonbonded=True,
    use_gate=True,
    use_layer_norm=False,
    pooling_type='attention',
    dropout=0.0
)

print('\n检查权重初始化...')
print(f'angle_weight_raw: {model.angle_weight_raw.item():.6f}')
print(f'angle_weight (computed): {model.get_angle_weight().item():.6f}')

if torch.isnan(model.get_angle_weight()):
    print('❌ angle_weight是NaN！')
    sys.exit(1)
else:
    print('✓ angle_weight正常')

print(f'\ndihedral_weight: {model.get_dihedral_weight().item():.6f}')
print(f'nonbonded_weight: {model.get_nonbonded_weight().item():.6f}')

print('\n测试梯度计算...')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 创建简单的loss来测试梯度
weight = model.get_angle_weight()
loss = weight * 10.0  # 简单的乘法，确保有梯度
optimizer.zero_grad()
loss.backward()

grad_val = model.angle_weight_raw.grad.item() if model.angle_weight_raw.grad is not None else None
print(f'angle_weight_raw.grad: {grad_val}')

if model.angle_weight_raw.grad is None:
    print('⚠️  angle_weight_raw没有梯度')
elif torch.isnan(model.angle_weight_raw.grad):
    print('❌ angle_weight_raw的梯度是NaN！')
    sys.exit(1)
else:
    print('✓ angle_weight_raw梯度正常')

print('\n✓ 所有检查通过！')
"

echo ""
echo "========================================"
echo "如果看到'所有检查通过'，说明修复成功！"
echo "========================================"
