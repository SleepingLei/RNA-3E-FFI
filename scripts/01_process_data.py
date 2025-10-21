#!/usr/bin/env python3
"""
Data Acquisition and Preprocessing Pipeline

This script orchestrates the preprocessing of RNA-ligand complexes from HARIBOSS:
1. Download HARIBOSS complex list
2. Download PDB structures
3. Define and extract binding pockets
4. Process with AmberTools
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
import pandas as pd
import requests
from Bio.PDB import PDBList, MMCIFParser
import MDAnalysis as mda
from tqdm import tqdm


def download_hariboss_list(output_csv_path):
    """
    Download the complete list of RNA-ligand complexes from HARIBOSS.

    Args:
        output_csv_path: Path to save the downloaded CSV file

    Returns:
        DataFrame containing the HARIBOSS complex list
    """
    url = "http://hariboss.pasteur.cloud/complexes/"
    print(f"Downloading HARIBOSS complex list from {url}...")

    try:
        # Try to download the CSV directly
        csv_url = f"{url}complexes.csv"
        response = requests.get(csv_url, timeout=30)
        response.raise_for_status()

        # Save to file
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_csv_path, 'wb') as f:
            f.write(response.content)

        print(f"Successfully downloaded to {output_csv_path}")

        # Read and return the dataframe
        df = pd.read_csv(output_csv_path)
        return df

    except Exception as e:
        print(f"Error downloading HARIBOSS list: {e}")
        print("Please manually download the CSV from http://hariboss.pasteur.cloud/complexes/")
        raise


def download_pdb_files(pdb_id_list, output_dir):
    """
    Download PDB structures in mmCIF format.

    Args:
        pdb_id_list: List of PDB IDs to download
        output_dir: Directory to save downloaded mmCIF files

    Returns:
        Dictionary mapping PDB IDs to their downloaded file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdb_list = PDBList()
    downloaded_files = {}

    print(f"Downloading {len(pdb_id_list)} PDB structures...")

    for pdb_id in tqdm(pdb_id_list):
        try:
            # Download in mmCIF format
            file_path = pdb_list.retrieve_pdb_file(
                pdb_id,
                file_format='mmCif',
                pdir=str(output_path)
            )

            if file_path and os.path.exists(file_path):
                downloaded_files[pdb_id] = file_path
            else:
                print(f"Warning: Failed to download {pdb_id}")

        except Exception as e:
            print(f"Error downloading {pdb_id}: {e}")

    print(f"Successfully downloaded {len(downloaded_files)} structures")
    return downloaded_files


def define_and_save_pocket(cif_path, ligand_resname, pocket_cutoff, output_pdb_path):
    """
    Define RNA binding pocket around ligand and save to PDB file.

    Args:
        cif_path: Path to input mmCIF file
        ligand_resname: Residue name of the ligand
        pocket_cutoff: Distance cutoff in Angstroms for pocket definition
        output_pdb_path: Path to save the pocket PDB file

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

        # Select RNA atoms
        try:
            rna = u.select_atoms("nucleic")
            if len(rna) == 0:
                print(f"Warning: No RNA atoms found in structure")
                return False
        except Exception as e:
            print(f"Error selecting RNA atoms: {e}")
            return False

        # Define pocket: RNA atoms within cutoff distance of ligand
        pocket = rna.select_atoms(f"around {pocket_cutoff} group ligand", ligand=ligand)

        if len(pocket) == 0:
            print(f"Warning: No pocket atoms found within {pocket_cutoff} Å")
            return False

        # Create output directory if needed
        output_path = Path(output_pdb_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save pocket to PDB file
        pocket.write(str(output_pdb_path))

        print(f"Saved pocket with {len(pocket)} atoms to {output_pdb_path}")
        return True

    except Exception as e:
        print(f"Error processing {cif_path}: {e}")
        return False


def run_ambertools(pocket_pdb_path, amber_output_prefix):
    """
    Run AmberTools to generate topology and coordinate files.

    Args:
        pocket_pdb_path: Path to input pocket PDB file
        amber_output_prefix: Prefix for output files (will create .prmtop and .inpcrd)

    Returns:
        True if successful, False otherwise
    """
    try:
        pocket_path = Path(pocket_pdb_path)
        output_prefix = Path(amber_output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: Clean PDB with pdb4amber
        cleaned_pdb_path = output_prefix.parent / f"{output_prefix.stem}_cleaned.pdb"

        print(f"Running pdb4amber on {pocket_pdb_path}...")
        pdb4amber_cmd = [
            "pdb4amber",
            "-i", str(pocket_path),
            "-o", str(cleaned_pdb_path),
            "--nohyd"  # Don't add hydrogens
        ]

        result = subprocess.run(
            pdb4amber_cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"pdb4amber error: {result.stderr}")
            # Try to continue anyway
            cleaned_pdb_path = pocket_path

        # Step 2: Generate tleap input script
        tleap_script = f"""source leaprc.RNA.OL3
mol = loadpdb {cleaned_pdb_path}
saveamberparm mol {output_prefix}.prmtop {output_prefix}.inpcrd
quit
"""

        # Save tleap script to temporary file
        tleap_script_path = output_prefix.parent / f"{output_prefix.stem}_tleap.in"
        with open(tleap_script_path, 'w') as f:
            f.write(tleap_script)

        # Step 3: Run tleap
        print(f"Running tleap...")
        tleap_cmd = ["tleap", "-f", str(tleap_script_path)]

        result = subprocess.run(
            tleap_cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(output_prefix.parent)
        )

        # Check if output files were created
        prmtop_path = Path(f"{output_prefix}.prmtop")
        inpcrd_path = Path(f"{output_prefix}.inpcrd")

        if prmtop_path.exists() and inpcrd_path.exists():
            print(f"Successfully created {prmtop_path} and {inpcrd_path}")
            # Clean up temporary files
            if tleap_script_path.exists():
                tleap_script_path.unlink()
            return True
        else:
            print(f"tleap failed to create output files")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"AmberTools timeout for {pocket_pdb_path}")
        return False
    except FileNotFoundError as e:
        print(f"AmberTools not found. Please ensure AmberTools is installed and in PATH: {e}")
        return False
    except Exception as e:
        print(f"Error running AmberTools on {pocket_pdb_path}: {e}")
        return False


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Process RNA-ligand complexes from HARIBOSS")
    parser.add_argument("--hariboss_csv", type=str, default="hariboss/Complexes.csv",
                        help="Path to HARIBOSS complexes CSV file")
    parser.add_argument("--download_list", action="store_true",
                        help="Download HARIBOSS list from web")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Base output directory")
    parser.add_argument("--pocket_cutoff", type=float, default=5.0,
                        help="Cutoff distance for pocket definition in Angstroms")
    parser.add_argument("--max_complexes", type=int, default=None,
                        help="Maximum number of complexes to process (for testing)")

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(args.output_dir)
    mmcif_dir = base_dir / "raw" / "mmCIF"
    pocket_dir = base_dir / "processed" / "pockets"
    amber_dir = base_dir / "processed" / "amber"

    # Create directories
    for dir_path in [mmcif_dir, pocket_dir, amber_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Download or read HARIBOSS list
    if args.download_list:
        hariboss_df = download_hariboss_list(args.hariboss_csv)
    else:
        print(f"Reading HARIBOSS CSV from {args.hariboss_csv}...")
        hariboss_df = pd.read_csv(args.hariboss_csv)

    print(f"Found {len(hariboss_df)} complexes in HARIBOSS dataset")

    # Limit number of complexes if specified
    if args.max_complexes:
        hariboss_df = hariboss_df.head(args.max_complexes)
        print(f"Processing first {len(hariboss_df)} complexes")

    # Extract PDB IDs (assuming column name is 'pdb_id' or 'PDB_ID' or similar)
    # Adjust column name based on actual CSV structure
    pdb_id_column = None
    for col in ['pdb_id', 'PDB_ID', 'pdbid', 'PDBID', 'PDB']:
        if col in hariboss_df.columns:
            pdb_id_column = col
            break

    if pdb_id_column is None:
        print("Available columns:", hariboss_df.columns.tolist())
        print("Error: Could not find PDB ID column in CSV")
        sys.exit(1)

    # Extract ligand residue name column
    ligand_column = None
    for col in ['ligand', 'Ligand', 'ligand_resname', 'LIGAND', 'ligand_name']:
        if col in hariboss_df.columns:
            ligand_column = col
            break

    if ligand_column is None:
        print("Warning: Could not find ligand column, using default 'LIG'")
        hariboss_df['ligand_resname'] = 'LIG'
        ligand_column = 'ligand_resname'

    # Process each complex
    success_count = 0
    failed_complexes = []

    print("\nProcessing complexes...")
    for idx, row in tqdm(hariboss_df.iterrows(), total=len(hariboss_df)):
        pdb_id = str(row[pdb_id_column]).lower()
        ligand_resname = str(row[ligand_column])

        print(f"\n--- Processing {pdb_id} (ligand: {ligand_resname}) ---")

        try:
            # Define file paths
            complex_id = f"{pdb_id}_{ligand_resname}"

            # Check if mmCIF file already exists, otherwise download
            cif_path = mmcif_dir / f"{pdb_id}.cif"
            if not cif_path.exists():
                print(f"Downloading {pdb_id}...")
                downloaded = download_pdb_files([pdb_id], mmcif_dir)
                if pdb_id in downloaded:
                    cif_path = Path(downloaded[pdb_id])
                else:
                    print(f"Failed to download {pdb_id}")
                    failed_complexes.append((pdb_id, "download_failed"))
                    continue

            pocket_pdb_path = pocket_dir / f"{complex_id}_pocket.pdb"
            amber_prefix = amber_dir / f"{complex_id}"

            # Step 1: Define and save pocket
            if not pocket_pdb_path.exists():
                print(f"Defining pocket for {complex_id}...")
                success = define_and_save_pocket(
                    cif_path,
                    ligand_resname,
                    args.pocket_cutoff,
                    pocket_pdb_path
                )
                if not success:
                    failed_complexes.append((pdb_id, "pocket_definition_failed"))
                    continue

            # Step 2: Run AmberTools
            prmtop_path = Path(f"{amber_prefix}.prmtop")
            if not prmtop_path.exists():
                print(f"Running AmberTools for {complex_id}...")
                success = run_ambertools(pocket_pdb_path, amber_prefix)
                if not success:
                    failed_complexes.append((pdb_id, "ambertools_failed"))
                    continue

            success_count += 1
            print(f"Successfully processed {complex_id}")

        except Exception as e:
            print(f"Unexpected error processing {pdb_id}: {e}")
            failed_complexes.append((pdb_id, str(e)))
            continue

    # Print summary
    print("\n" + "="*60)
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}/{len(hariboss_df)}")
    print(f"Failed: {len(failed_complexes)}")

    if failed_complexes:
        print("\nFailed complexes:")
        for pdb_id, reason in failed_complexes:
            print(f"  {pdb_id}: {reason}")

        # Save failed complexes to file
        failed_df = pd.DataFrame(failed_complexes, columns=['pdb_id', 'reason'])
        failed_path = base_dir / "failed_complexes.csv"
        failed_df.to_csv(failed_path, index=False)
        print(f"\nFailed complexes saved to {failed_path}")


if __name__ == "__main__":
    main()
