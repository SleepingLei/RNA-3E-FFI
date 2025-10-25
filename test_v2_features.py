#!/usr/bin/env python3
"""
Test script for v2.0 feature encoding with integer indices
"""
import sys
from pathlib import Path
import numpy as np
import torch

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from amber_vocabulary import get_global_encoder

# Import from 03_build_dataset
import importlib.util
spec = importlib.util.spec_from_file_location("build_dataset", Path(__file__).parent / "scripts" / "03_build_dataset.py")
build_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_dataset)
build_graph_from_files = build_dataset.build_graph_from_files

def test_encoder():
    """Test the integer-based feature encoder."""
    print("=" * 80)
    print("Testing Integer-based Feature Encoder")
    print("=" * 80)

    encoder = get_global_encoder()
    print(f"\n{encoder}")

    # Test encoding
    test_features = encoder.encode_atom_features(
        atom_type="C4",
        charge=-0.123,
        residue_name="G",
        atomic_number=6
    )

    print(f"\nTest 1: Standard atom (C4, charge=-0.123, residue=G, atomic_num=6)")
    print(f"  Feature shape: {test_features.shape}")
    print(f"  Feature values: {test_features}")
    print(f"  Expected: [atom_type_idx, charge, residue_idx, atomic_num]")
    print(f"  Feature dim: {encoder.feature_dim}")

    # Test unknown handling
    test_unknown = encoder.encode_atom_features(
        atom_type="UNKNOWN_TYPE",
        charge=0.5,
        residue_name="XYZ",
        atomic_number=99
    )

    print(f"\nTest 2: Unknown atom (UNKNOWN_TYPE, residue=XYZ)")
    print(f"  Feature shape: {test_unknown.shape}")
    print(f"  Feature values: {test_unknown}")
    print(f"  Note: Unknown indices should use <UNK> token")

    return encoder

def test_graph_construction():
    """Test graph construction with new features."""
    print("\n" + "=" * 80)
    print("Testing Graph Construction with Real Data")
    print("=" * 80)

    # Use the test output directory
    test_dir = Path("test_output/1aju_ARG_graph_intermediate")
    prmtop_path = test_dir / "rna.prmtop"
    inpcrd_path = test_dir / "rna.inpcrd"

    if not prmtop_path.exists():
        print(f"\n⚠️  Test file not found: {prmtop_path}")
        print("Skipping graph construction test.")
        return None

    print(f"\nInput files:")
    print(f"  PRMTOP: {prmtop_path}")
    print(f"  INPCRD: {inpcrd_path}")

    # Build graph
    data = build_graph_from_files(
        rna_pdb_path="dummy.pdb",  # Not used
        prmtop_path=str(prmtop_path),
        distance_cutoff=5.0,
        add_nonbonded_edges=True
    )

    if data is None:
        print("\n❌ Graph construction failed!")
        return None

    print(f"\n✅ Graph constructed successfully!")
    print(f"\nGraph structure:")
    print(f"  Nodes: {data.x.shape[0]}")
    print(f"  Node feature dim: {data.x.shape[1]} (should be 4)")
    print(f"  Positions: {data.pos.shape}")
    print(f"  1-hop edges (bonds): {data.edge_index.shape[1]}")
    print(f"  2-hop paths (angles): {data.triple_index.shape[1]}")
    print(f"  3-hop paths (dihedrals): {data.quadra_index.shape[1]}")
    print(f"  Non-bonded edges: {data.nonbonded_edge_index.shape[1]}")

    print(f"\nEdge attributes:")
    print(f"  Bond attr shape: {data.edge_attr.shape} (should be [num_bonds, 2])")
    print(f"  Angle attr shape: {data.triple_attr.shape} (should be [num_angles, 2])")
    print(f"  Dihedral attr shape: {data.quadra_attr.shape} (should be [num_dihedrals, 3])")
    print(f"  Non-bonded attr shape: {data.nonbonded_edge_attr.shape} (should be [num_nb, 3])")

    # Check node features
    print(f"\nSample node features (first 5 atoms):")
    print(f"  Format: [atom_type_idx, charge, residue_idx, atomic_num]")
    for i in range(min(5, data.x.shape[0])):
        print(f"  Atom {i}: {data.x[i].numpy()}")

    # Check LJ parameters
    print(f"\nSample LJ parameters (first 5 non-bonded edges):")
    print(f"  Format: [LJ_A, LJ_B, distance]")
    for i in range(min(5, data.nonbonded_edge_attr.shape[0])):
        lj_params = data.nonbonded_edge_attr[i].numpy()
        print(f"  Edge {i}: LJ_A={lj_params[0]:.4e}, LJ_B={lj_params[1]:.4e}, dist={lj_params[2]:.3f}Å")

    # Verify LJ parameters are not all zeros
    lj_A_values = data.nonbonded_edge_attr[:, 0].numpy()
    lj_B_values = data.nonbonded_edge_attr[:, 1].numpy()

    print(f"\nLJ parameter statistics:")
    print(f"  LJ_A: min={lj_A_values.min():.4e}, max={lj_A_values.max():.4e}, mean={lj_A_values.mean():.4e}")
    print(f"  LJ_B: min={lj_B_values.min():.4e}, max={lj_B_values.max():.4e}, mean={lj_B_values.mean():.4e}")

    if lj_A_values.sum() == 0 and lj_B_values.sum() == 0:
        print("  ⚠️  Warning: All LJ parameters are zero! Real extraction may have failed.")
    else:
        print("  ✅ LJ parameters successfully extracted from prmtop!")

    return data

def save_vocabularies(encoder):
    """Save vocabulary files."""
    print("\n" + "=" * 80)
    print("Saving Vocabularies")
    print("=" * 80)

    output_dir = Path("data/vocabularies")
    encoder.save_vocabularies(str(output_dir))

    print(f"\n✅ Vocabularies saved to: {output_dir}")
    print(f"  Files created:")
    print(f"    - atom_type_vocab.json ({len(encoder.atom_type_vocab)} types)")
    print(f"    - residue_vocab.json ({len(encoder.residue_vocab)} types)")

def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RNA-3E-FFI v2.0 Feature Encoding Test Suite")
    print("=" * 80)

    # Test 1: Encoder
    encoder = test_encoder()

    # Test 2: Graph construction
    data = test_graph_construction()

    # Test 3: Save vocabularies
    save_vocabularies(encoder)

    print("\n" + "=" * 80)
    print("✅ All Tests Completed!")
    print("=" * 80)
    print("\nSummary:")
    print(f"  ✅ Feature dimension: {encoder.feature_dim}")
    print(f"  ✅ Atom type vocabulary: {len(encoder.atom_type_vocab)} types")
    print(f"  ✅ Residue vocabulary: {len(encoder.residue_vocab)} types")
    if data is not None:
        print(f"  ✅ Graph construction: {data.x.shape[0]} atoms, {data.edge_index.shape[1]} edges")
        print(f"  ✅ LJ parameters: {'extracted' if data.nonbonded_edge_attr[:, 0].sum() > 0 else 'placeholder'}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
