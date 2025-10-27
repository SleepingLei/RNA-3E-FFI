#!/usr/bin/env python3
"""
Debug script to identify the source of CUDA index out-of-bounds error.
Run this with: CUDA_LAUNCH_BLOCKING=1 python scripts/debug_index_error.py
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.amber_vocabulary import get_global_encoder

def check_vocabularies():
    """Check vocabulary configuration."""
    print("="*70)
    print("VOCABULARY CHECK")
    print("="*70)

    # Load vocabulary files
    atom_vocab_path = Path("data/vocabularies/atom_type_vocab.json")
    residue_vocab_path = Path("data/vocabularies/residue_vocab.json")

    with open(atom_vocab_path) as f:
        atom_data = json.load(f)
    with open(residue_vocab_path) as f:
        residue_data = json.load(f)

    print(f"\nAtom vocabulary:")
    print(f"  - vocab size: {atom_data['num_types']}")
    print(f"  - unk_idx: {atom_data['unk_idx']}")
    print(f"  - max vocab idx: {max([int(k) for k in atom_data['idx_to_vocab'].keys()])}")

    print(f"\nResidue vocabulary:")
    print(f"  - vocab size: {residue_data['num_types']}")
    print(f"  - unk_idx: {residue_data['unk_idx']}")
    print(f"  - max vocab idx: {max([int(k) for k in residue_data['idx_to_vocab'].keys()])}")

    # Check encoder
    encoder = get_global_encoder()
    print(f"\nEncoder configuration:")
    print(f"  - num_atom_types: {encoder.num_atom_types}")
    print(f"  - num_residues: {encoder.num_residues}")

    return encoder

def check_data_files(encoder, max_check=10):
    """Check a few data files for index ranges."""
    print("\n" + "="*70)
    print("DATA FILE CHECK")
    print("="*70)

    # Load splits
    splits_path = Path("data/splits/splits.json")
    with open(splits_path) as f:
        splits = json.load(f)

    train_ids = splits['train']
    print(f"\nChecking first {max_check} training samples...")

    checked = 0
    for complex_id in train_ids:
        if checked >= max_check:
            break

        graph_path = Path(f"data/processed/graphs/{complex_id}.pt")

        if not graph_path.exists():
            continue

        try:
            data = torch.load(graph_path, weights_only=False)
            x = data.x

            atom_type_idx = x[:, 0].long()
            residue_idx = x[:, 2].long()

            print(f"\n{complex_id}:")
            print(f"  - atom_type range: [{atom_type_idx.min()}, {atom_type_idx.max()}]")
            print(f"  - residue range: [{residue_idx.min()}, {residue_idx.max()}]")
            print(f"  - num atoms: {x.shape[0]}")

            # Check for potential issues
            if atom_type_idx.max() > encoder.num_atom_types:
                print(f"  ⚠️  WARNING: atom_type_idx {atom_type_idx.max()} > {encoder.num_atom_types}")
            if residue_idx.max() > encoder.num_residues:
                print(f"  ⚠️  WARNING: residue_idx {residue_idx.max()} > {encoder.num_residues}")
            if atom_type_idx.min() < 0:
                print(f"  ⚠️  WARNING: negative atom_type_idx: {atom_type_idx.min()}")
            if residue_idx.min() < 0:
                print(f"  ⚠️  WARNING: negative residue_idx: {residue_idx.min()}")

            checked += 1

        except Exception as e:
            print(f"\n❌ Error loading {complex_id}: {e}")
            continue

    return checked

def test_model_forward():
    """Test model forward pass to trigger the actual error."""
    print("\n" + "="*70)
    print("MODEL FORWARD TEST")
    print("="*70)

    # Import dataset
    sys.path.insert(0, 'scripts')
    from scripts.amber_vocabulary import get_global_encoder

    # Load a sample data file
    splits_path = Path("data/splits/splits.json")
    with open(splits_path) as f:
        splits = json.load(f)

    train_ids = splits['train']

    # Find first valid file
    sample_data = None
    sample_id = None
    for complex_id in train_ids[:50]:
        graph_path = Path(f"data/processed/graphs/{complex_id}.pt")
        if graph_path.exists():
            sample_data = torch.load(graph_path, weights_only=False)
            sample_id = complex_id
            break

    if sample_data is None:
        print("❌ No valid data file found")
        return

    print(f"\nTesting with sample: {sample_id}")

    # Get encoder
    encoder = get_global_encoder()

    # Initialize model
    from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2

    model = RNAPocketEncoderV2(
        num_atom_types=encoder.num_atom_types,
        num_residues=encoder.num_residues,
        atom_embed_dim=64,
        residue_embed_dim=32,
        hidden_irreps="64x0e + 32x1o + 16x2e",
        output_dim=512,
        num_layers=3,
        use_multi_hop=True,
        use_nonbonded=True,
        pooling_type='attention'
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = model.to(device)
    sample_data = sample_data.to(device)

    print(f"\nModel embedding sizes:")
    print(f"  - atom_type_embedding: {model.input_embedding.atom_type_embedding.num_embeddings} (valid indices: 0-{model.input_embedding.atom_type_embedding.num_embeddings-1})")
    print(f"  - residue_embedding: {model.input_embedding.residue_embedding.num_embeddings} (valid indices: 0-{model.input_embedding.residue_embedding.num_embeddings-1})")

    print(f"\nInput data:")
    print(f"  - x.shape: {sample_data.x.shape}")
    print(f"  - atom_type range: [{sample_data.x[:, 0].min()}, {sample_data.x[:, 0].max()}]")
    print(f"  - residue range: [{sample_data.x[:, 2].min()}, {sample_data.x[:, 2].max()}]")

    # Detailed check before forward pass
    atom_idx = sample_data.x[:, 0].long()
    residue_idx = sample_data.x[:, 2].long()
    max_atom_embed = model.input_embedding.atom_type_embedding.num_embeddings
    max_residue_embed = model.input_embedding.residue_embedding.num_embeddings

    if (atom_idx >= max_atom_embed).any() or (atom_idx < 0).any():
        print("\n⚠️  WARNING: Atom indices out of bounds BEFORE forward pass!")
        bad_mask = (atom_idx >= max_atom_embed) | (atom_idx < 0)
        print(f"  - {bad_mask.sum()} atoms with invalid indices")
        print(f"  - Invalid indices: {atom_idx[bad_mask].tolist()[:10]}")

    if (residue_idx >= max_residue_embed).any() or (residue_idx < 0).any():
        print("\n⚠️  WARNING: Residue indices out of bounds BEFORE forward pass!")
        bad_mask = (residue_idx >= max_residue_embed) | (residue_idx < 0)
        print(f"  - {bad_mask.sum()} atoms with invalid indices")
        print(f"  - Invalid indices: {residue_idx[bad_mask].tolist()[:10]}")

    # Try forward pass
    try:
        print("\nAttempting forward pass...")
        with torch.no_grad():
            output = model(sample_data)
        print(f"✅ Forward pass successful! Output shape: {output.shape}")
    except RuntimeError as e:
        print(f"\n❌ Forward pass failed with error:")
        print(f"   {str(e)}")

        # Print more debug info
        print("\nDetailed index check:")
        x = sample_data.x
        atom_idx = x[:, 0].long()
        residue_idx = x[:, 2].long()

        # Check which atoms have out-of-bound indices
        max_atom_embed = model.input_embedding.atom_type_embedding.num_embeddings
        max_residue_embed = model.input_embedding.residue_embedding.num_embeddings

        bad_atom_mask = atom_idx >= max_atom_embed
        bad_residue_mask = residue_idx >= max_residue_embed

        if bad_atom_mask.any():
            bad_indices = atom_idx[bad_atom_mask]
            print(f"  - Found {bad_atom_mask.sum()} atoms with out-of-bound atom_type:")
            print(f"    Indices: {bad_indices.tolist()[:10]}")
            print(f"    Max allowed: {max_atom_embed - 1}")

        if bad_residue_mask.any():
            bad_indices = residue_idx[bad_residue_mask]
            print(f"  - Found {bad_residue_mask.sum()} atoms with out-of-bound residue:")
            print(f"    Indices: {bad_indices.tolist()[:10]}")
            print(f"    Max allowed: {max_residue_embed - 1}")

        return False

    return True

def main():
    """Main debug routine."""
    print("\n" + "="*70)
    print("RNA-3E-FFI INDEX ERROR DEBUGGER")
    print("="*70)

    # Step 1: Check vocabularies
    encoder = check_vocabularies()

    # Step 2: Check data files
    checked = check_data_files(encoder, max_check=10)

    if checked == 0:
        print("\n❌ No data files found to check!")
        print("   Make sure you're running this on the remote server where the data is located.")
        return

    # Step 3: Test model forward pass
    success = test_model_forward()

    print("\n" + "="*70)
    if success:
        print("✅ All checks passed!")
    else:
        print("❌ Error found - see details above")
    print("="*70)

if __name__ == "__main__":
    main()
