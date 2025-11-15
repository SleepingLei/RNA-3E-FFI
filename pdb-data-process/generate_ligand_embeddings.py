#!/usr/bin/env python3
"""
Generate ligand embeddings using Uni-Mol2 from SMILES with pH correction.
Based on scripts/02_embed_ligands.py
"""
import os
import sys
from pathlib import Path
import argparse
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm


def generate_embeddings_from_csv(csv_file, output_h5_path, batch_size=32):
    """
    Generate embeddings from SMILES CSV file using Uni-Mol.

    Args:
        csv_file: Path to CSV with columns: filename, molecule_name, smiles, corrected_smiles
        output_h5_path: Path to save embeddings HDF5 file
        batch_size: Batch size for processing
    """
    print(f"Loading SMILES from {csv_file}...")
    df = pd.read_csv(csv_file)

    if 'corrected_smiles' not in df.columns:
        print("Error: CSV must contain 'corrected_smiles' column")
        return

    # Filter out invalid SMILES
    valid_mask = (df['corrected_smiles'] != 'N/A') & (~df['corrected_smiles'].isna())
    df = df[valid_mask].reset_index(drop=True)

    print(f"Found {len(df)} valid ligands")

    if len(df) == 0:
        print("No valid ligands to process!")
        return

    # Prepare SMILES and IDs
    smiles_list = df['corrected_smiles'].tolist()
    # Use filename as ID (remove _ligands.pdb suffix)
    ids = [fname.replace('_ligands.pdb', '') for fname in df['filename']]

    print(f"\nGenerating embeddings using Uni-Mol...")

    try:
        from unimol_tools import UniMolRepr

        print("Initializing Uni-Mol model...")
        clf = UniMolRepr(
            data_type='molecule',
            remove_hs=False,
            model_name='unimolv2',
            model_size='1.1B',
            compute_atomic_reprs=False
        )

        embeddings_dict = {}

        # Process in batches
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]

            print(f"Processing batch {i//batch_size + 1}/{(len(smiles_list)-1)//batch_size + 1}...")

            try:
                reprs = clf.get_repr(batch_smiles, return_atomic_reprs=False)

                if isinstance(reprs, dict) and 'cls_repr' in reprs:
                    cls_reprs = reprs['cls_repr']
                else:
                    cls_reprs = reprs

                for j, (complex_id, embedding) in enumerate(zip(batch_ids, cls_reprs)):
                    if isinstance(embedding, np.ndarray):
                        embeddings_dict[complex_id] = embedding
                    else:
                        embeddings_dict[complex_id] = np.array(embedding)

            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                import traceback
                traceback.print_exc()

        if not embeddings_dict:
            print("No embeddings generated!")
            return

        # Normalize embeddings
        print(f"\nNormalizing {len(embeddings_dict)} embeddings...")
        all_embeddings = np.array(list(embeddings_dict.values()))
        embedding_mean = np.mean(all_embeddings, axis=0, keepdims=True)
        embedding_std = np.std(all_embeddings, axis=0, keepdims=True)
        embedding_std = np.where(embedding_std < 1e-8, 1.0, embedding_std)

        # Save normalization parameters
        output_path = Path(output_h5_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        norm_params_path = output_path.parent / "ligand_embedding_norm_params.npz"
        np.savez(norm_params_path,
                 mean=embedding_mean.squeeze(),
                 std=embedding_std.squeeze())
        print(f"Saved normalization parameters to {norm_params_path}")

        # Apply normalization
        for complex_id in embeddings_dict:
            embeddings_dict[complex_id] = (embeddings_dict[complex_id] - embedding_mean.squeeze()) / embedding_std.squeeze()

        # Save to HDF5
        print(f"\nSaving {len(embeddings_dict)} normalized embeddings to {output_h5_path}...")

        with h5py.File(output_h5_path, 'w') as f:
            for complex_id, embedding in embeddings_dict.items():
                f.create_dataset(complex_id, data=embedding)

        print(f"Successfully saved embeddings to {output_h5_path}")
        print(f"\nEmbedding shape: {list(embeddings_dict.values())[0].shape}")

    except ImportError:
        print("Error: unimol_tools not installed!")
        print("Install with: pip install unimol_tools")
        return
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Generate ligand embeddings from SMILES')
    parser.add_argument('--csv-file', default='ligands_smiles.csv',
                        help='Input CSV file with SMILES')
    parser.add_argument('--output-h5', default='ligand_embeddings.h5',
                        help='Output HDF5 file for embeddings')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing')

    args = parser.parse_args()

    generate_embeddings_from_csv(
        args.csv_file,
        args.output_h5,
        args.batch_size
    )


if __name__ == '__main__':
    main()
