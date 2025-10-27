#!/usr/bin/env python3
"""
Fix out-of-bound indices in processed data files.
This script will clamp indices to valid ranges.
"""
import torch
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.amber_vocabulary import get_global_encoder

def fix_data_files():
    """Fix all data files with out-of-bound indices."""

    # Get encoder and valid ranges
    encoder = get_global_encoder()

    # Expected ranges (1-indexed, 0 is padding)
    max_atom_idx = encoder.num_atom_types  # e.g., 70
    max_residue_idx = encoder.num_residues  # e.g., 43

    print("="*70)
    print("DATA INDEX FIXER")
    print("="*70)
    print(f"\nValid ranges:")
    print(f"  - atom_type: 0-{max_atom_idx} (0=padding, 1-{max_atom_idx}=valid)")
    print(f"  - residue: 0-{max_residue_idx} (0=padding, 1-{max_residue_idx}=valid)")

    # Load splits
    splits_path = Path("data/splits/splits.json")
    with open(splits_path) as f:
        splits = json.load(f)

    # Process all splits
    all_ids = splits['train'] + splits['val'] + splits['test']

    print(f"\nProcessing {len(all_ids)} data files...")

    fixed_count = 0
    error_count = 0
    skipped_count = 0

    for complex_id in tqdm(all_ids, desc="Checking files"):
        graph_path = Path(f"data/processed_v2/{complex_id}_pocket_graph.pt")

        if not graph_path.exists():
            skipped_count += 1
            continue

        try:
            data = torch.load(graph_path, weights_only=False)

            # Extract features
            x = data.x.clone()
            atom_type_idx = x[:, 0]
            residue_idx = x[:, 2]

            # Check for out-of-bound
            needs_fix = False

            # Fix atom types
            if (atom_type_idx > max_atom_idx).any():
                print(f"\n❌ {complex_id}: atom_type out of bound")
                print(f"   Range: [{atom_type_idx.min()}, {atom_type_idx.max()}]")
                print(f"   Clamping to max: {max_atom_idx}")

                # Clamp to max valid index
                x[:, 0] = torch.clamp(atom_type_idx, min=0, max=max_atom_idx)
                needs_fix = True

            if (atom_type_idx < 0).any():
                print(f"\n❌ {complex_id}: negative atom_type")
                print(f"   Setting negative indices to 0 (padding)")
                x[:, 0] = torch.clamp(atom_type_idx, min=0)
                needs_fix = True

            # Fix residues
            if (residue_idx > max_residue_idx).any():
                print(f"\n❌ {complex_id}: residue out of bound")
                print(f"   Range: [{residue_idx.min()}, {residue_idx.max()}]")
                print(f"   Clamping to max: {max_residue_idx}")

                x[:, 2] = torch.clamp(residue_idx, min=0, max=max_residue_idx)
                needs_fix = True

            if (residue_idx < 0).any():
                print(f"\n❌ {complex_id}: negative residue")
                print(f"   Setting negative indices to 0 (padding)")
                x[:, 2] = torch.clamp(residue_idx, min=0)
                needs_fix = True

            # Save if fixed
            if needs_fix:
                data.x = x
                torch.save(data, graph_path)
                fixed_count += 1

        except Exception as e:
            print(f"\n❌ Error processing {complex_id}: {e}")
            error_count += 1
            continue

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total files: {len(all_ids)}")
    print(f"Fixed: {fixed_count}")
    print(f"Errors: {error_count}")
    print(f"Skipped (not found): {skipped_count}")
    print(f"OK: {len(all_ids) - fixed_count - error_count - skipped_count}")

    if fixed_count > 0:
        print(f"\n⚠️  {fixed_count} files were modified!")
        print("    You should consider re-running data preprocessing instead of using this fix.")

if __name__ == "__main__":
    fix_data_files()
