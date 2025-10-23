#!/usr/bin/env python3
"""
Utility script to inspect HDF5 files (ligand embeddings)

This script helps you explore the contents of HDF5 files containing
ligand embeddings or other numerical data.
"""
import h5py
import numpy as np
import sys
from pathlib import Path


def print_h5_structure(h5_file, max_items=10):
    """
    Print the structure of an HDF5 file.

    Args:
        h5_file: Open h5py.File object
        max_items: Maximum number of items to display
    """
    print("=" * 70)
    print(f"HDF5 File Structure")
    print("=" * 70)

    keys = list(h5_file.keys())
    print(f"\nTotal number of keys: {len(keys)}")

    if len(keys) > 0:
        print(f"\nFirst {min(max_items, len(keys))} keys:")
        for i, key in enumerate(keys[:max_items]):
            dataset = h5_file[key]
            print(f"  [{i+1}] {key}")
            print(f"      Shape: {dataset.shape}")
            print(f"      Dtype: {dataset.dtype}")
            print(f"      Size: {dataset.size} elements")

            # Show first few values if 1D or small
            if len(dataset.shape) == 1 and dataset.shape[0] <= 10:
                print(f"      Values: {dataset[:]}")
            elif len(dataset.shape) <= 2 and dataset.size <= 50:
                print(f"      Sample:\n{dataset[:]}")
            else:
                print(f"      Sample (first 5 elements): {dataset[:].flat[:5]}")
            print()

        if len(keys) > max_items:
            print(f"  ... and {len(keys) - max_items} more keys")


def get_key_info(h5_file, key):
    """
    Get detailed information about a specific key.

    Args:
        h5_file: Open h5py.File object
        key: Key to inspect
    """
    if key not in h5_file:
        print(f"Error: Key '{key}' not found in file")
        return

    dataset = h5_file[key]

    print("=" * 70)
    print(f"Detailed Information for Key: '{key}'")
    print("=" * 70)
    print(f"\nShape: {dataset.shape}")
    print(f"Dtype: {dataset.dtype}")
    print(f"Size: {dataset.size} elements")
    print(f"Memory: {dataset.size * dataset.dtype.itemsize / 1024:.2f} KB")

    # Load data
    data = dataset[:]

    print(f"\nStatistics:")
    print(f"  Min: {np.min(data):.6f}")
    print(f"  Max: {np.max(data):.6f}")
    print(f"  Mean: {np.mean(data):.6f}")
    print(f"  Std: {np.std(data):.6f}")

    print(f"\nFirst 10 elements:")
    print(data.flat[:10])

    print(f"\nLast 10 elements:")
    print(data.flat[-10:])

    if len(data.shape) == 2:
        print(f"\nFirst row:")
        print(data[0])


def search_keys(h5_file, pattern):
    """
    Search for keys matching a pattern.

    Args:
        h5_file: Open h5py.File object
        pattern: String pattern to match (case-insensitive)
    """
    pattern = pattern.lower()
    keys = list(h5_file.keys())
    matches = [k for k in keys if pattern in k.lower()]

    print(f"\nFound {len(matches)} keys matching '{pattern}':")
    for key in matches:
        dataset = h5_file[key]
        print(f"  {key} - Shape: {dataset.shape}, Dtype: {dataset.dtype}")


def compare_embeddings(h5_file, key1, key2):
    """
    Compare two embeddings.

    Args:
        h5_file: Open h5py.File object
        key1: First key
        key2: Second key
    """
    if key1 not in h5_file or key2 not in h5_file:
        print("Error: One or both keys not found")
        return

    emb1 = h5_file[key1][:]
    emb2 = h5_file[key2][:]

    print(f"\nComparing '{key1}' vs '{key2}':")
    print(f"  Shape: {emb1.shape} vs {emb2.shape}")

    # Cosine similarity
    if emb1.shape == emb2.shape:
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)
        similarity = np.dot(emb1_norm.flat, emb2_norm.flat)
        print(f"  Cosine similarity: {similarity:.6f}")

        # Euclidean distance
        distance = np.linalg.norm(emb1 - emb2)
        print(f"  Euclidean distance: {distance:.6f}")


def interactive_mode(h5_path):
    """
    Interactive exploration mode.

    Args:
        h5_path: Path to HDF5 file
    """
    with h5py.File(h5_path, 'r') as f:
        print_h5_structure(f, max_items=20)

        print("\n" + "=" * 70)
        print("Interactive Mode - Available Commands:")
        print("=" * 70)
        print("  info <key>           - Show detailed info for a key")
        print("  search <pattern>     - Search for keys matching pattern")
        print("  compare <key1> <key2> - Compare two embeddings")
        print("  list                 - List all keys")
        print("  quit                 - Exit")
        print()

        while True:
            try:
                cmd = input("\nCommand: ").strip()

                if cmd == 'quit' or cmd == 'exit' or cmd == 'q':
                    break
                elif cmd == 'list':
                    print(f"\nAll keys ({len(f.keys())} total):")
                    for i, key in enumerate(f.keys(), 1):
                        print(f"  {i}. {key}")
                elif cmd.startswith('info '):
                    key = cmd[5:].strip()
                    get_key_info(f, key)
                elif cmd.startswith('search '):
                    pattern = cmd[7:].strip()
                    search_keys(f, pattern)
                elif cmd.startswith('compare '):
                    parts = cmd[8:].strip().split()
                    if len(parts) == 2:
                        compare_embeddings(f, parts[0], parts[1])
                    else:
                        print("Usage: compare <key1> <key2>")
                else:
                    print("Unknown command. Try: info, search, compare, list, or quit")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python inspect_h5.py <h5_file_path> [options]")
        print("\nOptions:")
        print("  --interactive    Start interactive mode")
        print("  --key <key>      Show info for specific key")
        print("  --search <pat>   Search for keys matching pattern")
        print("\nExamples:")
        print("  python inspect_h5.py data/processed/ligand_embeddings.h5")
        print("  python inspect_h5.py data/processed/ligand_embeddings.h5 --interactive")
        print("  python inspect_h5.py data/processed/ligand_embeddings.h5 --key 1aju_ARG")
        sys.exit(1)

    h5_path = Path(sys.argv[1])

    if not h5_path.exists():
        print(f"Error: File not found: {h5_path}")
        sys.exit(1)

    print(f"Opening HDF5 file: {h5_path}")
    print(f"File size: {h5_path.stat().st_size / 1024 / 1024:.2f} MB\n")

    # Check for options
    if '--interactive' in sys.argv:
        interactive_mode(h5_path)
    elif '--key' in sys.argv:
        key_idx = sys.argv.index('--key') + 1
        if key_idx < len(sys.argv):
            key = sys.argv[key_idx]
            with h5py.File(h5_path, 'r') as f:
                get_key_info(f, key)
    elif '--search' in sys.argv:
        search_idx = sys.argv.index('--search') + 1
        if search_idx < len(sys.argv):
            pattern = sys.argv[search_idx]
            with h5py.File(h5_path, 'r') as f:
                search_keys(f, pattern)
    else:
        # Default: just show structure
        with h5py.File(h5_path, 'r') as f:
            print_h5_structure(f, max_items=20)


if __name__ == "__main__":
    main()
