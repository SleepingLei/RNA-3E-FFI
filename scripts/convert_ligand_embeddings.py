#!/usr/bin/env python3
"""
Convert ligand embeddings from 'pdbid_ligandid' format to 'ligandid' format.
Each ligand ID will only have one embedding (duplicates are handled by a strategy).

Usage:
    python scripts/convert_ligand_embeddings.py <input.h5> <output.h5> [--strategy first|last|mean]
"""
import h5py
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict


def extract_ligand_id(key):
    """
    Extract ligand ID from 'pdbid_ligandid' format.

    Args:
        key: String in format 'pdbid_ligandid', e.g., '1aju_ARG'

    Returns:
        ligand_id: String like 'ARG'
    """
    parts = key.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[1:])  # Handle cases like 'pdbid_LIG_AND'
    return key


def convert_embeddings(input_path, output_path, strategy='first'):
    """
    Convert embeddings from pdbid_ligandid to ligandid format.

    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
        strategy: How to handle duplicates - 'first', 'last', or 'mean'
    """
    print(f"Reading input file: {input_path}")

    # Group embeddings by ligand ID
    ligand_embeddings = defaultdict(list)

    with h5py.File(input_path, 'r') as f_in:
        print(f"Total keys in input: {len(f_in.keys())}")

        for key in f_in.keys():
            ligand_id = extract_ligand_id(key)
            embedding = f_in[key][:]
            ligand_embeddings[ligand_id].append((key, embedding))

    print(f"\nUnique ligand IDs: {len(ligand_embeddings)}")

    # Show statistics about duplicates
    duplicates = {lid: len(embs) for lid, embs in ligand_embeddings.items() if len(embs) > 1}
    if duplicates:
        print(f"\nLigands with multiple embeddings: {len(duplicates)}")
        print("Top 10 ligands by occurrence:")
        sorted_dups = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)
        for lid, count in sorted_dups[:10]:
            pdb_ids = [key.split('_')[0] for key, _ in ligand_embeddings[lid]]
            print(f"  {lid}: {count} occurrences (PDBs: {', '.join(pdb_ids[:5])}{'...' if count > 5 else ''})")

    # Process embeddings based on strategy
    print(f"\nApplying strategy: {strategy}")
    final_embeddings = {}

    for ligand_id, embeddings in ligand_embeddings.items():
        if strategy == 'first':
            # Keep the first occurrence
            final_embeddings[ligand_id] = embeddings[0][1]
        elif strategy == 'last':
            # Keep the last occurrence
            final_embeddings[ligand_id] = embeddings[-1][1]
        elif strategy == 'mean':
            # Average all embeddings for this ligand
            emb_array = np.array([emb for _, emb in embeddings])
            final_embeddings[ligand_id] = np.mean(emb_array, axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # Write to output file
    print(f"\nWriting output file: {output_path}")
    with h5py.File(output_path, 'w') as f_out:
        for ligand_id, embedding in final_embeddings.items():
            f_out.create_dataset(ligand_id, data=embedding, dtype=np.float32)

    print(f"\nConversion complete!")
    print(f"Output file contains {len(final_embeddings)} unique ligands")

    # Verify output
    output_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"Output file size: {output_size:.2f} MB")


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python convert_ligand_embeddings.py <input.h5> <output.h5> [--strategy first|last|mean]")
        print("\nStrategies:")
        print("  first  - Keep the first occurrence of each ligand (default)")
        print("  last   - Keep the last occurrence of each ligand")
        print("  mean   - Average all embeddings for each ligand")
        print("\nExample:")
        print("  python scripts/convert_ligand_embeddings.py data/processed/ligand_embeddings.h5 data/processed/ligand_unique.h5 --strategy first")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    # Parse strategy
    strategy = 'first'
    if '--strategy' in sys.argv:
        strategy_idx = sys.argv.index('--strategy') + 1
        if strategy_idx < len(sys.argv):
            strategy = sys.argv[strategy_idx]
            if strategy not in ['first', 'last', 'mean']:
                print(f"Error: Invalid strategy '{strategy}'. Must be 'first', 'last', or 'mean'")
                sys.exit(1)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if output_path.exists():
        response = input(f"Output file {output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    convert_embeddings(input_path, output_path, strategy)


if __name__ == "__main__":
    main()
