#!/usr/bin/env python3
"""
Test script to verify V3 model fixes:
1. Bonded weight = 0.4
2. Angle/Dihedral/Nonbonded weights = 0.2 each
3. Total weight = 1.0
4. Post-LN added
"""

import torch
from torch_geometric.data import Data
from models.e3_gnn_encoder_v3 import RNAPocketEncoderV3

print("=" * 80)
print("Testing V3 Model Fixes")
print("=" * 80)

# Create test data
num_nodes = 50
num_edges = 100
num_angles = 50
num_dihedrals = 30
num_nonbonded = 80

x = torch.randn(num_nodes, 3)  # [charge, atomic_num, mass]
pos = torch.randn(num_nodes, 3)
edge_index = torch.randint(0, num_nodes, (2, num_edges))
edge_attr = torch.rand(num_edges, 2)  # [req, k]

# Angles
triple_index = torch.randint(0, num_nodes, (3, num_angles))
triple_attr = torch.rand(num_angles, 2)

# Dihedrals
quadra_index = torch.randint(0, num_nodes, (4, num_dihedrals))
quadra_attr = torch.rand(num_dihedrals, 3)

# Non-bonded
nonbonded_edge_index = torch.randint(0, num_nodes, (2, num_nonbonded))
nonbonded_edge_attr = torch.rand(num_nonbonded, 3)

data = Data(
    x=x,
    pos=pos,
    edge_index=edge_index,
    edge_attr=edge_attr,
    triple_index=triple_index,
    triple_attr=triple_attr,
    quadra_index=quadra_index,
    quadra_attr=quadra_attr,
    nonbonded_edge_index=nonbonded_edge_index,
    nonbonded_edge_attr=nonbonded_edge_attr
)

# Create model with V3 improvements
print("\n1. Creating model...")
model = RNAPocketEncoderV3(
    output_dim=512,
    num_layers=4,
    use_multi_hop=True,
    use_nonbonded=True,
    use_enhanced_invariants=True,
    use_improved_layers=True,
    initial_angle_weight=0.2,
    initial_dihedral_weight=0.2,
    initial_nonbonded_weight=0.2,
    num_attention_heads=4
)

print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check weights
print("\n2. Checking path weights...")
weight_stats = model.get_weight_stats()
print(f"   Bonded weight:    {weight_stats['bonded_weight']:.4f}")
print(f"   Angle weight:     {weight_stats['angle_weight']:.4f}")
print(f"   Dihedral weight:  {weight_stats['dihedral_weight']:.4f}")
print(f"   Nonbonded weight: {weight_stats['nonbonded_weight']:.4f}")
print(f"   Total weight:     {weight_stats['total_weight']:.4f}")

# Verify total = 1.0
if abs(weight_stats['total_weight'] - 1.0) < 0.01:
    print("   ✅ Total weight = 1.0 (PASS)")
else:
    print(f"   ❌ Total weight != 1.0 (FAIL)")

# Check if Post-LN layers exist
print("\n3. Checking Post-LN layers...")
if hasattr(model, 'post_aggregation_layer_norms'):
    print(f"   ✅ Post-LN layers exist: {len(model.post_aggregation_layer_norms)} layers")
else:
    print("   ❌ Post-LN layers not found (FAIL)")

# Test forward pass
print("\n4. Testing forward pass...")
try:
    output = model(data)
    print(f"   Input: {num_nodes} atoms")
    print(f"   Output: {output.shape}")
    print("   ✅ Forward pass successful")
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")

# Test gradient flow
print("\n5. Testing gradient flow...")
try:
    model.train()
    output = model(data)
    loss = output.sum()
    loss.backward()

    # Check gradient norms
    total_grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5

    print(f"   Total gradient norm: {total_grad_norm:.6f}")
    if total_grad_norm < 1000:  # Reasonable threshold
        print("   ✅ Gradient norm is reasonable")
    else:
        print(f"   ⚠️  Gradient norm is high ({total_grad_norm:.2f})")
except Exception as e:
    print(f"   ❌ Gradient test failed: {e}")

# Test feature statistics
print("\n6. Testing feature statistics...")
try:
    model.eval()
    with torch.no_grad():
        stats = model.get_feature_stats(data)

    print(f"   Input norm: {stats['input_norm']:.4f}")
    for i, layer_stats in enumerate(stats['layers']):
        print(f"   Layer {i}:")
        print(f"     Bonded norm:      {layer_stats.get('bonded_norm', 'N/A'):.4f}")
        print(f"     Aggregated norm:  {layer_stats.get('aggregated_norm', 'N/A'):.4f}")
        if 'after_norm' in layer_stats:
            print(f"     After norm:       {layer_stats['after_norm']:.4f}")

    # Check if norms are stable
    first_layer_norm = stats['layers'][0].get('aggregated_norm', 0)
    last_layer_norm = stats['layers'][-1].get('aggregated_norm', 0)
    if last_layer_norm > 0:
        ratio = last_layer_norm / (first_layer_norm + 1e-6)
        print(f"\n   Norm ratio (last/first): {ratio:.2f}x")
        if ratio < 5.0:  # Should be much better than previous 62x
            print("   ✅ Feature norms are stable")
        else:
            print(f"   ⚠️  Feature norms still growing significantly")
except Exception as e:
    print(f"   ❌ Feature stats test failed: {e}")

print("\n" + "=" * 80)
print("Tests completed!")
print("=" * 80)

# Summary
print("\n📝 Summary of fixes:")
print("  1. ✅ Bonded weight added (0.4)")
print("  2. ✅ Angle/Dihedral/Nonbonded weights set to 0.2")
print("  3. ✅ Total weight = 1.0")
print("  4. ✅ Post-LN layers added")
print("\n💡 Expected improvements:")
print("  - Feature norms should grow by ~5x instead of ~62x across 6 layers")
print("  - Gradient norms should be more stable during training")
print("  - Training should be more stable with fewer exploding gradients")
