#!/usr/bin/env python3
"""
Ligand Embedding Generation using Uni-Mol2

This script extracts ligands from RNA-ligand complexes and generates
3D-aware embeddings using Uni-Mol.
"""
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from unimol_tools import UniMolRepr


def extract_ligand_to_sdf(cif_path, ligand_resname, output_sdf_path):
    """
    Extract ligand from mmCIF file and save to SDF format with 3D coordinates.

    Args:
        cif_path: Path to input mmCIF file
        ligand_resname: Residue name of the ligand
        output_sdf_path: Path to save the ligand SDF file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load structure with MDAnalysis
        u = mda.Universe(cif_path)

        # Select ligand atoms
        try:
            ligand = u.select_atoms(f"resname {ligand_resname}")
            if len(ligand) == 0:
                print(f"Warning: No ligand found with resname {ligand_resname}")
                return False
        except Exception as e:
            print(f"Error selecting ligand: {e}")
            return False

        # Create output directory if needed
        output_path = Path(output_sdf_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write ligand to temporary PDB
        temp_pdb = output_path.parent / f"{output_path.stem}_temp.pdb"
        ligand.write(str(temp_pdb))

        # Convert PDB to RDKit molecule preserving 3D coordinates
        mol = Chem.MolFromPDBFile(str(temp_pdb), removeHs=False, sanitize=False)

        if mol is None:
            print(f"Warning: RDKit failed to parse ligand from {temp_pdb}")
            # Try without sanitize=False
            mol = Chem.MolFromPDBFile(str(temp_pdb), removeHs=False)

        if mol is None:
            print(f"Error: Could not convert ligand to RDKit molecule")
            return False

        # Try to sanitize the molecule
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"Warning: Sanitization failed: {e}")
            # Continue anyway - Uni-Mol might still work

        # Get 3D coordinates from conformer
        if mol.GetNumConformers() == 0:
            print(f"Warning: No conformer found, using MDAnalysis coordinates")
            # Create conformer from MDAnalysis coordinates
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i, pos in enumerate(ligand.positions):
                conf.SetAtomPosition(i, tuple(pos))
            mol.AddConformer(conf)

        # Write to SDF file
        writer = Chem.SDWriter(str(output_sdf_path))
        writer.write(mol)
        writer.close()

        # Clean up temporary file
        if temp_pdb.exists():
            temp_pdb.unlink()

        print(f"Saved ligand to {output_sdf_path}")
        return True

    except Exception as e:
        print(f"Error extracting ligand from {cif_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_ligand_embeddings(sdf_paths, complex_ids, output_h5_path, batch_size=32):
    """
    Generate embeddings for all ligands using Uni-Mol.

    Args:
        sdf_paths: List of paths to ligand SDF files
        complex_ids: List of complex IDs corresponding to each SDF
        output_h5_path: Path to save embeddings HDF5 file
        batch_size: Batch size for processing

    Returns:
        Dictionary mapping complex IDs to embedding vectors
    """
    print(f"\nGenerating embeddings for {len(sdf_paths)} ligands using Uni-Mol...")

    try:
        # Initialize Uni-Mol representation model
        print("Initializing Uni-Mol model...")
        clf = UniMolRepr(data_type='molecule', remove_hs=False,model_name='unimolv2', model_size='1.1B',compute_atomic_reprs=False)

        # Process all ligands
        embeddings_dict = {}

        # Process in batches
        for i in range(0, len(sdf_paths), batch_size):
            batch_paths = sdf_paths[i:i+batch_size]
            batch_ids = complex_ids[i:i+batch_size]

            print(f"Processing batch {i//batch_size + 1}/{(len(sdf_paths)-1)//batch_size + 1}...")

            try:
                # Get representations
                # Uni-Mol expects a list of molecule data
                reprs = clf.get_repr(batch_paths, return_atomic_reprs=False)

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

        # Save embeddings to HDF5 file
        print(f"\nSaving {len(embeddings_dict)} embeddings to {output_h5_path}...")
        output_path = Path(output_h5_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

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
    parser = argparse.ArgumentParser(description="Generate ligand embeddings using Uni-Mol")
    parser.add_argument("--hariboss_csv", type=str, default="hariboss/Complexes.csv",
                        help="Path to HARIBOSS complexes CSV file")
    parser.add_argument("--mmcif_dir", type=str, default="data/raw/mmCIF",
                        help="Directory containing mmCIF files")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Output directory for SDF files and embeddings")
    parser.add_argument("--output_h5", type=str, default="data/processed/ligand_embeddings.h5",
                        help="Output HDF5 file for embeddings")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for Uni-Mol processing")
    parser.add_argument("--max_complexes", type=int, default=None,
                        help="Maximum number of complexes to process (for testing)")

    args = parser.parse_args()

    # Setup paths
    mmcif_dir = Path(args.mmcif_dir)
    sdf_dir = Path(args.output_dir) / "ligands"
    sdf_dir.mkdir(parents=True, exist_ok=True)

    # Read HARIBOSS list
    print(f"Reading HARIBOSS CSV from {args.hariboss_csv}...")
    hariboss_df = pd.read_csv(args.hariboss_csv)
    print(f"Found {len(hariboss_df)} complexes")

    # Limit number of complexes if specified
    if args.max_complexes:
        hariboss_df = hariboss_df.head(args.max_complexes)
        print(f"Processing first {len(hariboss_df)} complexes")

    # Find PDB ID and ligand columns
    pdb_id_column = None
    for col in ['pdb_id', 'PDB_ID', 'pdbid', 'PDBID', 'PDB']:
        if col in hariboss_df.columns:
            pdb_id_column = col
            break

    if pdb_id_column is None:
        print("Available columns:", hariboss_df.columns.tolist())
        print("Error: Could not find PDB ID column in CSV")
        sys.exit(1)

    ligand_column = None
    for col in ['ligand', 'Ligand', 'ligand_resname', 'LIGAND', 'ligand_name']:
        if col in hariboss_df.columns:
            ligand_column = col
            break

    if ligand_column is None:
        print("Warning: Could not find ligand column, using default 'LIG'")
        hariboss_df['ligand_resname'] = 'LIG'
        ligand_column = 'ligand_resname'

    # Extract ligands to SDF files
    sdf_paths = []
    complex_ids = []
    failed_extractions = []

    print("\nExtracting ligands to SDF files...")
    for idx, row in tqdm(hariboss_df.iterrows(), total=len(hariboss_df)):
        pdb_id = str(row[pdb_id_column]).lower()
        ligand_resname = str(row[ligand_column])
        complex_id = f"{pdb_id}_{ligand_resname}"

        # Define file paths
        cif_path = mmcif_dir / f"{pdb_id}.cif"
        sdf_path = sdf_dir / f"{complex_id}.sdf"

        # Skip if SDF already exists
        if sdf_path.exists():
            sdf_paths.append(str(sdf_path))
            complex_ids.append(complex_id)
            continue

        # Check if mmCIF file exists
        if not cif_path.exists():
            print(f"Warning: mmCIF file not found for {pdb_id}")
            failed_extractions.append((complex_id, "mmcif_not_found"))
            continue

        # Extract ligand
        try:
            success = extract_ligand_to_sdf(cif_path, ligand_resname, sdf_path)
            if success:
                sdf_paths.append(str(sdf_path))
                complex_ids.append(complex_id)
            else:
                failed_extractions.append((complex_id, "extraction_failed"))
        except Exception as e:
            print(f"Error extracting {complex_id}: {e}")
            failed_extractions.append((complex_id, str(e)))

    print(f"\nSuccessfully extracted {len(sdf_paths)} ligands")
    print(f"Failed extractions: {len(failed_extractions)}")

    if failed_extractions:
        failed_df = pd.DataFrame(failed_extractions, columns=['complex_id', 'reason'])
        failed_path = Path(args.output_dir) / "failed_ligand_extractions.csv"
        failed_df.to_csv(failed_path, index=False)
        print(f"Failed extractions saved to {failed_path}")

    # Generate embeddings
    if len(sdf_paths) > 0:
        embeddings = generate_ligand_embeddings(
            sdf_paths,
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
