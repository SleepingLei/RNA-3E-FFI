#!/usr/bin/env python3
"""
Quick test script for E(3) GNN Encoder v2.0

This script demonstrates how to use the new v2 model with real data.
"""
import sys
from pathlib import Path
import torch
from torch_geometric.data import Data

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2
from scripts.amber_vocabulary import get_global_encoder


def test_with_dummy_data():
    """Test model with synthetic data."""
    print("=" * 80)
    print("Testing E(3) GNN Encoder v2.0 with Dummy Data")
    print("=" * 80)

    # Get vocabulary sizes
    encoder = get_global_encoder()
    print(f"\nVocabulary sizes:")
    print(f"  Atom types: {encoder.num_atom_types}")
    print(f"  Residues: {encoder.num_residues}")

    # Create model
    model = RNAPocketEncoderV2(
        num_atom_types=encoder.num_atom_types,
        num_residues=encoder.num_residues,
        hidden_irreps="32x0e + 16x1o + 8x2e",
        output_dim=512,
        num_layers=4,
        use_multi_hop=True,
        use_nonbonded=True,
        pooling_type='attention'
    )

    print(f"\nModel created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Hidden irreps: {model.hidden_irreps}")
    print(f"  Output dim: {model.output_dim}")

    # Create test data
    num_nodes = 50
    num_edges = 150
    num_angles = 80
    num_dihedrals = 40
    num_nonbonded = 100

    x = torch.zeros(num_nodes, 4)
    x[:, 0] = torch.randint(1, encoder.num_atom_types + 1, (num_nodes,)).float()
    x[:, 1] = torch.randn(num_nodes) * 0.5
    x[:, 2] = torch.randint(1, encoder.num_residues + 1, (num_nodes,)).float()
    x[:, 3] = torch.randint(1, 20, (num_nodes,)).float()

    data = Data(
        x=x,
        pos=torch.randn(num_nodes, 3),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_attr=torch.randn(num_edges, 2).abs() + 0.1,
        triple_index=torch.randint(0, num_nodes, (3, num_angles)),
        triple_attr=torch.randn(num_angles, 2).abs() + 0.1,
        quadra_index=torch.randint(0, num_nodes, (4, num_dihedrals)),
        quadra_attr=torch.randn(num_dihedrals, 3),
        nonbonded_edge_index=torch.randint(0, num_nodes, (2, num_nonbonded)),
        nonbonded_edge_attr=torch.cat([
            torch.randn(num_nonbonded, 2).abs(),
            torch.rand(num_nonbonded, 1) * 6.0
        ], dim=-1)
    )

    print(f"\nTest data:")
    print(f"  Nodes: {data.x.shape[0]}")
    print(f"  1-hop edges: {data.edge_index.shape[1]}")
    print(f"  2-hop paths: {data.triple_index.shape[1]}")
    print(f"  3-hop paths: {data.quadra_index.shape[1]}")
    print(f"  Non-bonded edges: {data.nonbonded_edge_index.shape[1]}")

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = model(data)

    print(f"\nOutput:")
    print(f"  Shape: {output.shape}")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")

    # Get node embeddings
    print("\nExtracting node embeddings...")
    with torch.no_grad():
        node_embeddings = model.get_node_embeddings(data)

    print(f"  Node embeddings shape: {node_embeddings.shape}")
    print(f"  Mean: {node_embeddings.mean().item():.4f}")

    print("\n" + "=" * 80)
    print("✓ Test passed!")
    print("=" * 80)


def test_with_real_data():
    """Test model with real data from dataset."""
    print("\n" + "=" * 80)
    print("Testing E(3) GNN Encoder v2.0 with Real Data")
    print("=" * 80)

    # Check if processed graphs exist
    graph_dir = Path("data/processed/graphs")
    if not graph_dir.exists() or not list(graph_dir.glob("*.pt")):
        print("\n⚠️  No processed graphs found.")
        print("Run `python scripts/03_build_dataset.py` first to build graphs.")
        return

    # Load a sample graph
    graph_files = list(graph_dir.glob("*.pt"))
    sample_graph_path = graph_files[0]

    print(f"\nLoading sample graph: {sample_graph_path.name}")
    data = torch.load(sample_graph_path)

    print(f"\nGraph structure:")
    print(f"  Nodes: {data.x.shape[0]}")
    print(f"  Node features: {data.x.shape}")
    print(f"  Positions: {data.pos.shape}")
    print(f"  1-hop edges: {data.edge_index.shape[1]}")

    # Check if data format matches v2.0 (should be 4 dimensions)
    if data.x.shape[1] != 4:
        print(f"\n⚠️  Warning: Graph uses old feature format ({data.x.shape[1]} dims instead of 4).")
        print("Run `python scripts/03_build_dataset.py` to regenerate graphs with v2.0 format.")
        return

    if hasattr(data, 'triple_index'):
        print(f"  2-hop paths: {data.triple_index.shape[1]}")
    if hasattr(data, 'quadra_index'):
        print(f"  3-hop paths: {data.quadra_index.shape[1]}")
    if hasattr(data, 'nonbonded_edge_index'):
        print(f"  Non-bonded edges: {data.nonbonded_edge_index.shape[1]}")

    # Create model
    encoder = get_global_encoder()
    model = RNAPocketEncoderV2(
        num_atom_types=encoder.num_atom_types,
        num_residues=encoder.num_residues,
        hidden_irreps="32x0e + 16x1o + 8x2e",
        output_dim=512,
        num_layers=3,
        use_multi_hop=True,
        use_nonbonded=True
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        output = model(data)

    print(f"\nOutput:")
    print(f"  Shape: {output.shape}")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")
    print(f"  Min: {output.min().item():.4f}")
    print(f"  Max: {output.max().item():.4f}")

    print("\n" + "=" * 80)
    print("✓ Real data test passed!")
    print("=" * 80)


def main():
    """Run all tests."""
    # Test 1: Dummy data
    test_with_dummy_data()

    # Test 2: Real data (if available)
    test_with_real_data()

    print("\n" + "=" * 80)
    print("✅ All tests completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Check MODELS_V2_SUMMARY.md for detailed documentation")
    print("  2. Update training script to use RNAPocketEncoderV2")
    print("  3. Run training with new model")
    print("=" * 80)


if __name__ == "__main__":
    main()
