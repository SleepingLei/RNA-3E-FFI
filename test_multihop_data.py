#!/usr/bin/env python3
"""
Test script to verify multi-hop graph construction from prmtop files.
"""

import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import after path is set
import torch

# Import build_dataset module
import importlib.util
spec = importlib.util.spec_from_file_location("build_dataset", scripts_dir / "03_build_dataset.py")
build_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_dataset)
build_graph_from_files = build_dataset.build_graph_from_files

def test_multihop_graph():
    """Test multi-hop graph construction on the 1aju_ARG example."""

    print("=" * 80)
    print("Testing Multi-Hop Graph Construction")
    print("=" * 80)

    # Test files
    test_dir = Path("test_output/1aju_ARG_graph_intermediate")
    rna_pdb = test_dir / "rna_only.pdb"
    rna_prmtop = test_dir / "rna.prmtop"

    if not rna_pdb.exists() or not rna_prmtop.exists():
        print(f"Error: Test files not found in {test_dir}")
        print(f"  Looking for: {rna_pdb}")
        print(f"  Looking for: {rna_prmtop}")
        return False

    # Build graph
    print(f"\nBuilding graph from:")
    print(f"  PDB: {rna_pdb}")
    print(f"  PRMTOP: {rna_prmtop}")

    data = build_graph_from_files(
        rna_pdb_path=rna_pdb,
        prmtop_path=rna_prmtop,
        distance_cutoff=5.0,
        add_nonbonded_edges=True
    )

    if data is None:
        print("\nError: Graph construction failed!")
        return False

    # Print statistics
    print("\n" + "=" * 80)
    print("Graph Statistics")
    print("=" * 80)

    print(f"\nNode features:")
    print(f"  Shape: {data.x.shape}")
    print(f"  Feature dim: {data.x.shape[1]}")

    print(f"\nPositions:")
    print(f"  Shape: {data.pos.shape}")

    print(f"\n1-hop (Bonded edges):")
    print(f"  edge_index shape: {data.edge_index.shape}")
    print(f"  Number of edges: {data.edge_index.shape[1]}")
    if hasattr(data, 'edge_attr'):
        print(f"  edge_attr shape: {data.edge_attr.shape}")
        print(f"  Sample bond params: {data.edge_attr[0].tolist()}")

    print(f"\n2-hop (Angle paths):")
    print(f"  triple_index shape: {data.triple_index.shape}")
    print(f"  Number of angles: {data.triple_index.shape[1]}")
    if hasattr(data, 'triple_attr'):
        print(f"  triple_attr shape: {data.triple_attr.shape}")
        if data.triple_attr.shape[0] > 0:
            print(f"  Sample angle params: {data.triple_attr[0].tolist()}")

    print(f"\n3-hop (Dihedral paths):")
    print(f"  quadra_index shape: {data.quadra_index.shape}")
    print(f"  Number of dihedrals: {data.quadra_index.shape[1]}")
    if hasattr(data, 'quadra_attr'):
        print(f"  quadra_attr shape: {data.quadra_attr.shape}")
        if data.quadra_attr.shape[0] > 0:
            print(f"  Sample dihedral params: {data.quadra_attr[0].tolist()}")

    print(f"\nNon-bonded edges:")
    if hasattr(data, 'nonbonded_edge_index'):
        print(f"  nonbonded_edge_index shape: {data.nonbonded_edge_index.shape}")
        print(f"  Number of non-bonded edges: {data.nonbonded_edge_index.shape[1]}")
        if hasattr(data, 'nonbonded_edge_attr'):
            print(f"  nonbonded_edge_attr shape: {data.nonbonded_edge_attr.shape}")

    # Test model compatibility
    print("\n" + "=" * 80)
    print("Testing Model Compatibility")
    print("=" * 80)

    try:
        from models.e3_gnn_encoder import RNAPocketEncoder

        # Create model
        input_dim = data.x.shape[1]
        model = RNAPocketEncoder(
            input_dim=input_dim,
            hidden_irreps="16x0e + 8x1o + 4x2e",
            output_dim=128,
            num_layers=2,
            use_gate=True
        )

        print(f"\nCreated model with input_dim={input_dim}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Forward pass
        with torch.no_grad():
            output = model(data)

        print(f"\nForward pass successful!")
        print(f"  Input shape: {data.x.shape}")
        print(f"  Output shape: {output.shape}")

        print("\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\nError during model test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_multihop_graph()
    sys.exit(0 if success else 1)
