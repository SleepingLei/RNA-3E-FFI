#!/usr/bin/env python3
"""
Debug script to find out-of-bound indices in training data.
"""
import torch
import json
from pathlib import Path
from tqdm import tqdm

def check_data_indices():
    """Check all training data for out-of-bound indices."""

    # Load splits
    splits_path = Path("data/splits/splits.json")
    with open(splits_path) as f:
        splits = json.load(f)

    train_ids = splits['train']

    print(f"Checking {len(train_ids)} training samples...")

    # Expected vocabulary sizes (from model initialization)
    NUM_ATOM_TYPES = 70  # +1 for padding = 71 total embeddings (indices 0-70)
    NUM_RESIDUES = 43    # +1 for padding = 44 total embeddings (indices 0-43)

    max_atom_idx = 0
    max_residue_idx = 0

    problematic_samples = []

    for complex_id in tqdm(train_ids, desc="Checking indices"):
        graph_path = Path(f"data/processed/graphs/{complex_id}.pt")

        if not graph_path.exists():
            continue

        try:
            data = torch.load(graph_path, weights_only=False)

            # Extract features
            x = data.x  # [num_atoms, 4]
            atom_type_idx = x[:, 0].long()
            residue_idx = x[:, 2].long()

            # Check atom indices
            max_atom = atom_type_idx.max().item()
            min_atom = atom_type_idx.min().item()

            # Check residue indices
            max_residue = residue_idx.max().item()
            min_residue = residue_idx.min().item()

            # Update global max
            max_atom_idx = max(max_atom_idx, max_atom)
            max_residue_idx = max(max_residue_idx, max_residue)

            # Check for out-of-bound
            if max_atom > NUM_ATOM_TYPES:
                problematic_samples.append({
                    'id': complex_id,
                    'issue': 'atom_type',
                    'max_value': max_atom,
                    'expected_max': NUM_ATOM_TYPES,
                    'min_value': min_atom
                })
                print(f"\n❌ {complex_id}: atom_type_idx out of bound!")
                print(f"   Range: [{min_atom}, {max_atom}], Expected max: {NUM_ATOM_TYPES}")

            if max_residue > NUM_RESIDUES:
                problematic_samples.append({
                    'id': complex_id,
                    'issue': 'residue_idx',
                    'max_value': max_residue,
                    'expected_max': NUM_RESIDUES,
                    'min_value': min_residue
                })
                print(f"\n❌ {complex_id}: residue_idx out of bound!")
                print(f"   Range: [{min_residue}, {max_residue}], Expected max: {NUM_RESIDUES}")

            if min_atom < 0:
                print(f"\n❌ {complex_id}: negative atom_type_idx: {min_atom}")

            if min_residue < 0:
                print(f"\n❌ {complex_id}: negative residue_idx: {min_residue}")

        except Exception as e:
            print(f"\n❌ Error loading {complex_id}: {e}")
            continue

    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Maximum atom_type_idx found: {max_atom_idx} (embedding size: {NUM_ATOM_TYPES + 1}, valid range: 0-{NUM_ATOM_TYPES})")
    print(f"Maximum residue_idx found: {max_residue_idx} (embedding size: {NUM_RESIDUES + 1}, valid range: 0-{NUM_RESIDUES})")
    print(f"\nProblematic samples: {len(problematic_samples)}")

    if problematic_samples:
        print("\nFirst 10 problematic samples:")
        for sample in problematic_samples[:10]:
            print(f"  - {sample['id']}: {sample['issue']} = {sample['max_value']} (expected max: {sample['expected_max']})")

    return problematic_samples, max_atom_idx, max_residue_idx

if __name__ == "__main__":
    problematic, max_atom, max_residue = check_data_indices()

    if problematic:
        print(f"\n⚠️  Found {len(problematic)} samples with out-of-bound indices!")
        print("You need to either:")
        print("1. Fix the data preprocessing to ensure indices are within bounds")
        print("2. Increase the embedding vocabulary sizes in the model")
    else:
        print("\n✅ All indices are within bounds!")
