#!/usr/bin/env python3
"""
Deduplicate Ligand Embeddings by Ligand Name

This script processes ligand_embeddings.h5 to keep only one embedding per unique
ligand name. For example, from multiple entries like:
  - 2kx8_ARG_model2
  - 1uui_ARG_model0
  - 3xyz_ARG_model1

It will keep only one entry with key "ARG".

Strategy:
  - Extract ligand name from complex IDs (e.g., "2kx8_ARG_model2" -> "ARG")
  - For each ligand name, keep the first occurrence
  - Save to a new deduplicated HDF5 file

Usage:
    python scripts/deduplicate_ligand_embeddings.py \\
        --input data/processed/ligand_embeddings.h5 \\
        --output data/processed/ligand_embeddings_dedup.h5
"""
import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def extract_ligand_name(complex_id):
    """
    Extract ligand name from complex ID.

    Args:
        complex_id: e.g., "2kx8_ARG_model2" or "1uui_P12_model0"

    Returns:
        Ligand name, e.g., "ARG" or "P12"
    """
    # Remove model suffix if present
    if '_model' in complex_id:
        base = complex_id.split('_model')[0]
    else:
        base = complex_id

    # Split by underscore and take the ligand part (second element)
    # Format: <pdb_id>_<ligand_name>
    parts = base.split('_')

    if len(parts) >= 2:
        # Return the ligand name (everything after first underscore)
        return '_'.join(parts[1:])
    else:
        # Fallback: return as-is
        return base


def deduplicate_ligand_embeddings(input_path, output_path, strategy='first'):
    """
    Deduplicate ligand embeddings by ligand name.

    Args:
        input_path: Path to input HDF5 file with full complex IDs as keys
        output_path: Path to output HDF5 file with ligand names as keys
        strategy: How to handle duplicates ('first', 'last', 'mean')
    """
    print(f"Loading ligand embeddings from {input_path}...")

    # Read all embeddings and group by ligand name
    ligand_groups = defaultdict(list)

    with h5py.File(input_path, 'r') as f_in:
        print(f"Total entries in input file: {len(f_in.keys())}")

        for complex_id in tqdm(f_in.keys(), desc="Grouping by ligand name"):
            ligand_name = extract_ligand_name(complex_id)
            embedding = np.array(f_in[complex_id][:])
            ligand_groups[ligand_name].append({
                'complex_id': complex_id,
                'embedding': embedding
            })

    print(f"\nFound {len(ligand_groups)} unique ligand names")

    # Show statistics
    print("\nDuplication statistics:")
    duplicate_counts = [(name, len(entries)) for name, entries in ligand_groups.items()]
    duplicate_counts.sort(key=lambda x: x[1], reverse=True)

    print(f"  Total unique ligands: {len(ligand_groups)}")
    print(f"  Ligands with most duplicates (top 10):")
    for name, count in duplicate_counts[:10]:
        print(f"    {name}: {count} entries")

    # Apply deduplication strategy
    print(f"\nApplying deduplication strategy: {strategy}")
    deduplicated = {}

    for ligand_name, entries in tqdm(ligand_groups.items(), desc="Deduplicating"):
        if strategy == 'first':
            # Keep first occurrence
            deduplicated[ligand_name] = entries[0]['embedding']
        elif strategy == 'last':
            # Keep last occurrence
            deduplicated[ligand_name] = entries[-1]['embedding']
        elif strategy == 'mean':
            # Average all embeddings for this ligand
            embeddings = np.array([e['embedding'] for e in entries])
            deduplicated[ligand_name] = np.mean(embeddings, axis=0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # Write deduplicated embeddings
    print(f"\nWriting deduplicated embeddings to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f_out:
        for ligand_name, embedding in tqdm(deduplicated.items(), desc="Writing"):
            f_out.create_dataset(ligand_name, data=embedding)

    print(f"\n✓ Deduplication complete!")
    print(f"  Input:  {len(ligand_groups)} entries (with full complex IDs)")
    print(f"  Output: {len(deduplicated)} unique ligand names")
    print(f"  Saved to: {output_path}")

    # Show example mappings
    print("\nExample ligand name mappings:")
    for ligand_name, entries in list(ligand_groups.items())[:5]:
        print(f"  {ligand_name}:")
        for entry in entries[:3]:
            print(f"    - {entry['complex_id']}")
        if len(entries) > 3:
            print(f"    ... and {len(entries) - 3} more")


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate ligand embeddings by ligand name"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/ligand_embeddings.h5",
        help="Input HDF5 file with ligand embeddings"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/ligand_embeddings_dedup.h5",
        help="Output HDF5 file for deduplicated embeddings"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="first",
        choices=["first", "last", "mean"],
        help="Deduplication strategy: first (keep first), last (keep last), mean (average all)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    if output_path.exists():
        response = input(f"Output file {output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    deduplicate_ligand_embeddings(input_path, output_path, args.strategy)


if __name__ == "__main__":
    main()
