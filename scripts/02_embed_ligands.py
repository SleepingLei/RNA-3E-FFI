#!/usr/bin/env python3
"""
Ligand Embedding Generation using Uni-Mol2

This script generates embeddings for ligands using their SMILES representations
from the compounds database. SMILES are pH-adjusted using OpenBabel at pH 7.4.
"""
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import subprocess
import tempfile
import ast
from tqdm import tqdm
from unimol_tools import UniMolRepr


def process_smiles_with_obabel(smiles, ph=7.4):
    """
    Process SMILES with OpenBabel to adjust protonation state at given pH.

    Args:
        smiles: Input SMILES string
        ph: Target pH value (default: 7.4)

    Returns:
        pH-adjusted SMILES string, or None if processing fails
    """
    try:
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as f_in:
            f_in.write(smiles + '\n')
            temp_in = f_in.name

        with tempfile.NamedTemporaryFile(mode='r', suffix='.smi', delete=False) as f_out:
            temp_out = f_out.name

        # Run obabel with pH adjustment
        # -p sets the pH for protonation state
        cmd = ['obabel', temp_in, '-osmi', '-O', temp_out, '-p', str(ph)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            print(f"Warning: obabel failed for SMILES {smiles}: {result.stderr}")
            # Clean up
            os.unlink(temp_in)
            os.unlink(temp_out)
            return None

        # Read the processed SMILES
        with open(temp_out, 'r') as f:
            processed_smiles = f.read().strip().split()[0]  # First column is SMILES

        # Clean up temporary files
        os.unlink(temp_in)
        os.unlink(temp_out)

        return processed_smiles

    except Exception as e:
        print(f"Error processing SMILES with obabel: {e}")
        return None


def extract_ligand_id(ligand_ids_str):
    """
    Extract ligand ID from the sm_ligand_ids string.

    Args:
        ligand_ids_str: String representation of ligand IDs list, e.g., "['ARG_.:B/47:A']"

    Returns:
        Ligand ID (e.g., 'ARG') or None if extraction fails
    """
    try:
        # Parse the string as a Python list
        ligand_list = ast.literal_eval(ligand_ids_str)
        if not ligand_list:
            return None

        # Take the first ligand ID and extract the compound code
        # Format is like 'ARG_.:B/47:A', we want 'ARG'
        full_id = ligand_list[0]
        ligand_code = full_id.split('_')[0]
        return ligand_code

    except Exception as e:
        print(f"Error extracting ligand ID from {ligand_ids_str}: {e}")
        return None


def generate_ligand_embeddings(smiles_list, complex_ids, output_h5_path, batch_size=32):
    """
    Generate embeddings for all ligands using Uni-Mol from SMILES.

    Args:
        smiles_list: List of SMILES strings (pH-adjusted)
        complex_ids: List of complex IDs corresponding to each SMILES
        output_h5_path: Path to save embeddings HDF5 file
        batch_size: Batch size for processing

    Returns:
        Dictionary mapping complex IDs to embedding vectors
    """
    print(f"\nGenerating embeddings for {len(smiles_list)} ligands using Uni-Mol...")

    try:
        # Initialize Uni-Mol representation model
        print("Initializing Uni-Mol model...")
        clf = UniMolRepr(data_type='molecule', remove_hs=False, model_name='unimolv2',
                        model_size='1.1B', compute_atomic_reprs=False)

        # Process all ligands
        embeddings_dict = {}

        # Process in batches
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]
            batch_ids = complex_ids[i:i+batch_size]

            print(f"Processing batch {i//batch_size + 1}/{(len(smiles_list)-1)//batch_size + 1}...")

            try:
                # Get representations from SMILES
                # UniMol can accept SMILES strings directly
                reprs = clf.get_repr(batch_smiles, return_atomic_reprs=False)

                # Extract CLS representations
                # reprs is typically a dictionary with 'cls_repr' key
                if isinstance(reprs, dict) and 'cls_repr' in reprs:
                    cls_reprs = reprs['cls_repr']
                else:
                    cls_reprs = reprs

                # Store embeddings
                for j, (complex_id, embedding) in enumerate(zip(batch_ids, cls_reprs)):
                    if isinstance(embedding, np.ndarray):
                        embeddings_dict[complex_id] = embedding
                    else:
                        embeddings_dict[complex_id] = np.array(embedding)

            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next batch

        # Normalize all embeddings (z-score normalization across all embeddings)
        print(f"\nNormalizing {len(embeddings_dict)} embeddings...")
        all_embeddings = np.array(list(embeddings_dict.values()))
        embedding_mean = np.mean(all_embeddings, axis=0, keepdims=True)
        embedding_std = np.std(all_embeddings, axis=0, keepdims=True)
        # Add small epsilon to avoid division by zero
        embedding_std = np.where(embedding_std < 1e-8, 1.0, embedding_std)

        # Save normalization parameters
        output_path = Path(output_h5_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        norm_params_path = output_path.parent / "ligand_embedding_norm_params.npz"
        np.savez(norm_params_path,
                 mean=embedding_mean.squeeze(),
                 std=embedding_std.squeeze())
        print(f"Saved normalization parameters to {norm_params_path}")

        # Apply normalization to each embedding
        for complex_id in embeddings_dict:
            embeddings_dict[complex_id] = (embeddings_dict[complex_id] - embedding_mean.squeeze()) / embedding_std.squeeze()

        # Save embeddings to HDF5 file
        print(f"\nSaving {len(embeddings_dict)} normalized embeddings to {output_h5_path}...")

        with h5py.File(output_h5_path, 'w') as f:
            for complex_id, embedding in embeddings_dict.items():
                f.create_dataset(complex_id, data=embedding)

        print(f"Successfully saved embeddings to {output_h5_path}")
        return embeddings_dict

    except Exception as e:
        print(f"Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    """Main ligand embedding generation pipeline."""
    parser = argparse.ArgumentParser(description="Generate ligand embeddings using Uni-Mol from SMILES")
    parser.add_argument("--complexes_csv", type=str, default="hariboss/Complexes.csv",
                        help="Path to HARIBOSS complexes CSV file")
    parser.add_argument("--compounds_csv", type=str, default="hariboss/compounds.csv",
                        help="Path to compounds CSV file with SMILES")
    parser.add_argument("--output_dir", type=str, default="data/processed/ligands",
                        help="Output directory for processed data")
    parser.add_argument("--output_h5", type=str, default="data/processed/ligand_embeddings.h5",
                        help="Output HDF5 file for embeddings")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for Uni-Mol processing")
    parser.add_argument("--ph", type=float, default=7.4,
                        help="pH value for protonation state adjustment (default: 7.4)")
    parser.add_argument("--max_complexes", type=int, default=None,
                        help="Maximum number of complexes to process (for testing)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read HARIBOSS complexes
    print(f"Reading complexes from {args.complexes_csv}...")
    complexes_df = pd.read_csv(args.complexes_csv)
    print(f"Found {len(complexes_df)} complexes")

    # Read compounds database
    print(f"Reading compounds from {args.compounds_csv}...")
    compounds_df = pd.read_csv(args.compounds_csv)
    print(f"Found {len(compounds_df)} compounds")

    # Create a mapping from compound ID to SMILES
    compound_smiles_map = dict(zip(compounds_df['id'], compounds_df['canonical_smiles']))

    # Limit number of complexes if specified
    if args.max_complexes:
        complexes_df = complexes_df.head(args.max_complexes)
        print(f"Processing first {len(complexes_df)} complexes")

    # Find PDB ID column
    pdb_id_column = None
    for col in ['id', 'pdb_id', 'PDB_ID', 'pdbid', 'PDBID', 'PDB']:
        if col in complexes_df.columns:
            pdb_id_column = col
            break

    if pdb_id_column is None:
        print("Available columns:", complexes_df.columns.tolist())
        print("Error: Could not find PDB ID column in CSV")
        sys.exit(1)

    # Process ligands and prepare SMILES
    smiles_list = []
    complex_ids = []
    failed_processing = []

    print("\nProcessing ligands and adjusting pH...")
    for idx, row in tqdm(complexes_df.iterrows(), total=len(complexes_df)):
        pdb_id = str(row[pdb_id_column]).lower()
        ligand_ids_str = str(row['sm_ligand_ids'])

        # Extract ligand ID
        ligand_id = extract_ligand_id(ligand_ids_str)
        if ligand_id is None:
            failed_processing.append((pdb_id, "failed_to_extract_ligand_id"))
            continue

        complex_id = f"{pdb_id}_{ligand_id}"

        # Get SMILES from compounds database
        if ligand_id not in compound_smiles_map:
            failed_processing.append((complex_id, f"ligand_not_found_in_compounds_db"))
            continue

        original_smiles = compound_smiles_map[ligand_id]

        # Process SMILES with obabel to adjust pH

        processed_smiles = process_smiles_with_obabel(original_smiles, ph=args.ph)
        if processed_smiles is None:
            # If obabel fails, use original SMILES
            print(f"Warning: Using original SMILES for {complex_id}")
            processed_smiles = original_smiles

        smiles_list.append(processed_smiles)
        complex_ids.append(complex_id)

    print(f"\nSuccessfully processed {len(smiles_list)} ligands")
    print(f"Failed processing: {len(failed_processing)}")

    if failed_processing:
        failed_df = pd.DataFrame(failed_processing, columns=['complex_id', 'reason'])
        failed_path = output_dir / "failed_ligand_processing.csv"
        failed_df.to_csv(failed_path, index=False)
        print(f"Failed processing saved to {failed_path}")

    # Save processed SMILES for reference
    if len(smiles_list) > 0:
        smiles_df = pd.DataFrame({
            'complex_id': complex_ids,
            'smiles': smiles_list
        })
        smiles_path = output_dir / "processed_ligand_smiles.csv"
        smiles_df.to_csv(smiles_path, index=False)
        print(f"Processed SMILES saved to {smiles_path}")

        # Generate embeddings
        embeddings = generate_ligand_embeddings(
            smiles_list,
            complex_ids,
            args.output_h5,
            batch_size=args.batch_size
        )

        print(f"\n{'='*60}")
        print(f"Embedding generation complete!")
        print(f"Successfully generated {len(embeddings)} embeddings")
        print(f"Saved to {args.output_h5}")
    else:
        print("\nNo ligands to process!")


if __name__ == "__main__":
    main()
