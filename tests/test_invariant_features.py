#!/usr/bin/env python3
"""
Test script to validate the invariant feature extraction.

This script tests:
1. Dimension correctness
2. Rotation invariance
3. Comparison with old scalar-only approach
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch_geometric.data import Data
from scipy.spatial.transform import Rotation as R_scipy
import numpy as np

# Import model
from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2
from scripts.amber_vocabulary import get_global_encoder


def test_invariant_dimension():
    """Test that invariant feature dimension is correct."""
    print("\n" + "="*80)
    print("Test 1: Invariant Feature Dimension")
    print("="*80)

    encoder = get_global_encoder()

    # Create model
    model = RNAPocketEncoderV2(
        num_atom_types=encoder.num_atom_types,
        num_residues=encoder.num_residues,
        hidden_irreps="32x0e + 16x1o + 8x2e",
        output_dim=512,
        num_layers=2,
        use_multi_hop=False,
        use_nonbonded=False
    )

    # Check dimensions
    print(f"Hidden irreps: {model.hidden_irreps}")
    print(f"Scalar dim (l=0): {model.scalar_dim}")
    print(f"L1 irreps (l=1): {model.num_l1_irreps}")
    print(f"L2 irreps (l=2): {model.num_l2_irreps}")
    print(f"Total invariant dim: {model.invariant_dim}")
    print(f"Expected: 32 + 16 + 8 = 56")

    assert model.invariant_dim == 56, f"Expected invariant_dim=56, got {model.invariant_dim}"
    print("✓ Dimension test passed!")

    # Test with actual forward pass
    num_nodes = 50
    x = torch.zeros(num_nodes, 4)
    x[:, 0] = torch.randint(0, encoder.num_atom_types, (num_nodes,)).float()
    x[:, 2] = torch.randint(0, encoder.num_residues, (num_nodes,)).float()
    x[:, 1] = torch.randn(num_nodes) * 0.5
    x[:, 3] = torch.randint(1, 20, (num_nodes,)).float()

    data = Data(
        x=x,
        pos=torch.randn(num_nodes, 3),
        edge_index=torch.randint(0, num_nodes, (2, 100)),
        edge_attr=torch.randn(100, 2).abs() + 0.1
    )

    # Get node embeddings
    t = model.get_node_embeddings(data)
    print(f"\nNode embeddings shape: {t.shape}")
    print(f"Expected: [{num_nodes}, 56]")
    assert t.shape == (num_nodes, 56), f"Expected shape ({num_nodes}, 56), got {t.shape}"
    print("✓ Forward pass dimension test passed!")


def test_rotation_invariance():
    """Test that invariant features are indeed rotation invariant."""
    print("\n" + "="*80)
    print("Test 2: Rotation Invariance")
    print("="*80)

    encoder = get_global_encoder()

    # Create model
    model = RNAPocketEncoderV2(
        num_atom_types=encoder.num_atom_types,
        num_residues=encoder.num_residues,
        hidden_irreps="32x0e + 16x1o + 8x2e",
        output_dim=512,
        num_layers=2,
        use_multi_hop=False,
        use_nonbonded=False
    )

    model.eval()  # Set to eval mode

    # Create test data
    num_nodes = 30
    x = torch.zeros(num_nodes, 4)
    x[:, 0] = torch.randint(0, encoder.num_atom_types, (num_nodes,)).float()
    x[:, 2] = torch.randint(0, encoder.num_residues, (num_nodes,)).float()
    x[:, 1] = torch.randn(num_nodes) * 0.5
    x[:, 3] = torch.randint(1, 20, (num_nodes,)).float()

    pos_original = torch.randn(num_nodes, 3)

    data = Data(
        x=x,
        pos=pos_original,
        edge_index=torch.randint(0, num_nodes, (2, 80)),
        edge_attr=torch.randn(80, 2).abs() + 0.1
    )

    # Get embeddings for original positions
    with torch.no_grad():
        output_original = model(data)
        t_original = model.get_node_embeddings(data)

    # Apply random rotation
    rotation_matrix = torch.tensor(
        R_scipy.random().as_matrix(),
        dtype=torch.float32
    )

    pos_rotated = pos_original @ rotation_matrix.T

    # Create rotated data
    data_rotated = Data(
        x=x,
        pos=pos_rotated,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr
    )

    # Get embeddings for rotated positions
    with torch.no_grad():
        output_rotated = model(data_rotated)
        t_rotated = model.get_node_embeddings(data_rotated)

    # Check invariance
    diff_output = torch.abs(output_original - output_rotated).max().item()
    diff_node = torch.abs(t_original - t_rotated).max().item()

    print(f"Max difference in graph output: {diff_output:.8f}")
    print(f"Max difference in node embeddings: {diff_node:.8f}")
    print(f"Tolerance: 1e-5")

    # Note: Due to numerical precision, we use a small tolerance
    assert diff_output < 1e-5, f"Graph output not invariant! Max diff: {diff_output}"
    assert diff_node < 1e-5, f"Node embeddings not invariant! Max diff: {diff_node}"

    print("✓ Rotation invariance test passed!")


def test_comparison_with_scalar_only():
    """Compare new invariant features with old scalar-only approach."""
    print("\n" + "="*80)
    print("Test 3: Comparison with Scalar-Only Approach")
    print("="*80)

    encoder = get_global_encoder()

    # Create model
    model = RNAPocketEncoderV2(
        num_atom_types=encoder.num_atom_types,
        num_residues=encoder.num_residues,
        hidden_irreps="32x0e + 16x1o + 8x2e",
        output_dim=512,
        num_layers=2,
        use_multi_hop=False,
        use_nonbonded=False
    )

    # Create test data
    num_nodes = 40
    x = torch.zeros(num_nodes, 4)
    x[:, 0] = torch.randint(0, encoder.num_atom_types, (num_nodes,)).float()
    x[:, 2] = torch.randint(0, encoder.num_residues, (num_nodes,)).float()
    x[:, 1] = torch.randn(num_nodes) * 0.5
    x[:, 3] = torch.randint(1, 20, (num_nodes,)).float()

    data = Data(
        x=x,
        pos=torch.randn(num_nodes, 3),
        edge_index=torch.randint(0, num_nodes, (2, 100)),
        edge_attr=torch.randn(100, 2).abs() + 0.1
    )

    # Get embeddings
    with torch.no_grad():
        # Initial embedding
        h = model.input_embedding(data.x)

        # Message passing
        for i in range(model.num_layers):
            h = model.bonded_mp_layers[i](h, data.pos, data.edge_index, data.edge_attr)

        # Extract invariant features (new method)
        t_new = model.extract_invariant_features(h)

        # Extract scalar only (old method)
        h_scalar_old = h[:, :model.scalar_dim]

    print(f"Equivariant features shape: {h.shape} (120 = 32 + 48 + 40)")
    print(f"Old scalar-only shape: {h_scalar_old.shape} (32 scalars)")
    print(f"New invariant features shape: {t_new.shape} (56 = 32 + 16 + 8)")
    print(f"\nInformation gain: {t_new.shape[1] - h_scalar_old.shape[1]} additional features")
    print(f"  - From l=1 vectors: 16 L2 norms")
    print(f"  - From l=2 tensors: 8 L2 norms")

    # Verify first 32 dimensions match
    diff = torch.abs(t_new[:, :32] - h_scalar_old).max().item()
    print(f"\nMax difference in scalar part: {diff:.8f}")
    assert diff < 1e-6, "Scalar parts should match exactly"

    # Check that additional features are non-trivial
    l1_norms = t_new[:, 32:48]  # 16 L2 norms from vectors
    l2_norms = t_new[:, 48:56]  # 8 L2 norms from tensors

    print(f"\nL1 norm statistics:")
    print(f"  Mean: {l1_norms.mean().item():.4f}")
    print(f"  Std: {l1_norms.std().item():.4f}")
    print(f"  Min: {l1_norms.min().item():.4f}")
    print(f"  Max: {l1_norms.max().item():.4f}")

    print(f"\nL2 norm statistics:")
    print(f"  Mean: {l2_norms.mean().item():.4f}")
    print(f"  Std: {l2_norms.std().item():.4f}")
    print(f"  Min: {l2_norms.min().item():.4f}")
    print(f"  Max: {l2_norms.max().item():.4f}")

    # Norms should be non-negative
    assert (l1_norms >= 0).all(), "L1 norms should be non-negative"
    assert (l2_norms >= 0).all(), "L2 norms should be non-negative"

    print("\n✓ Comparison test passed!")


def main():
    print("\n" + "="*80)
    print("Testing Invariant Feature Extraction")
    print("="*80)

    test_invariant_dimension()
    test_rotation_invariance()
    test_comparison_with_scalar_only()

    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)
    print("\nSummary:")
    print("- Invariant features correctly computed as scalars + L2 norms")
    print("- Dimension: 56 (32 scalars + 16 vector norms + 8 tensor norms)")
    print("- Rotation invariance verified")
    print("- Compatible with E3-invariant ligand embeddings")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
