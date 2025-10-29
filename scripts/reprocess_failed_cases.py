#!/usr/bin/env python3
"""
Reprocess failed cases that have empty prmtop files.

This script:
1. Identifies all empty prmtop files
2. Deletes the empty prmtop files
3. Re-runs the parameterization for those cases using the fixed clean_rna_terminal_atoms function
"""
import sys
from pathlib import Path
import pandas as pd
import subprocess

def main():
    print("="*70)
    print("Reprocessing Failed Cases with Empty Prmtop Files")
    print("="*70)

    # Find all empty prmtop files
    amber_dir = Path("data/processed/amber")
    empty_prmtops = []

    print("\nScanning for empty prmtop files...")
    for prmtop_file in amber_dir.glob("*_rna.prmtop"):
        if prmtop_file.stat().st_size == 0:
            empty_prmtops.append(prmtop_file)

    print(f"Found {len(empty_prmtops)} empty prmtop files")

    if len(empty_prmtops) == 0:
        print("No empty prmtop files found. Nothing to do.")
        return

    # Extract complex IDs from failed cases
    failed_complexes = set()
    for prmtop_file in empty_prmtops:
        # e.g., "1f27_BTN_model0_rna.prmtop" -> "1f27_BTN_model0"
        name = prmtop_file.stem.replace("_rna", "")
        # Extract base complex ID: "1f27_BTN_model0" -> "1f27_BTN"
        if "_model" in name:
            complex_id = name.split("_model")[0]
        else:
            complex_id = name
        failed_complexes.add(complex_id)

    print(f"Unique complexes to reprocess: {len(failed_complexes)}")

    # Delete empty prmtop files
    print("\nDeleting empty prmtop files...")
    for prmtop_file in empty_prmtops:
        prmtop_file.unlink()
        # Also delete corresponding inpcrd file if exists
        inpcrd_file = prmtop_file.with_suffix('.inpcrd')
        if inpcrd_file.exists():
            inpcrd_file.unlink()
    print(f"Deleted {len(empty_prmtops)} empty prmtop files")

    # Create a temporary CSV with only failed cases
    print("\nCreating temporary CSV for failed cases...")
    hariboss_csv = Path("hariboss/Complexes.csv")
    hariboss_df = pd.read_csv(hariboss_csv)

    # Find PDB ID column
    pdb_col = None
    for col in ['id', 'pdb_id', 'PDB_ID']:
        if col in hariboss_df.columns:
            pdb_col = col
            break

    # Find ligand column
    ligand_col = None
    for col in ['sm_ligand_ids', 'ligand', 'ligand_resname']:
        if col in hariboss_df.columns:
            ligand_col = col
            break

    # Filter to only failed cases
    def matches_failed(row):
        pdb_id = str(row[pdb_col]).lower()
        ligand_str = str(row[ligand_col])

        # Parse ligand name
        if ligand_col == 'sm_ligand_ids':
            try:
                import ast
                ligands = ast.literal_eval(ligand_str)
                if not isinstance(ligands, list):
                    ligands = [ligand_str]
            except:
                ligands = [ligand_str]

            if ligands and len(ligands) > 0:
                ligand_resname = ligands[0].split('_')[0].split(':')[0]
            else:
                ligand_resname = 'LIG'
        else:
            ligand_resname = ligand_str

        complex_id = f"{pdb_id}_{ligand_resname}"
        return complex_id in failed_complexes

    failed_df = hariboss_df[hariboss_df.apply(matches_failed, axis=1)]

    print(f"Found {len(failed_df)} rows in HARIBOSS CSV matching failed cases")

    # Save temporary CSV
    temp_csv = Path("data/processed/temp_failed_cases.csv")
    failed_df.to_csv(temp_csv, index=False)
    print(f"Saved to {temp_csv}")

    # Run 01_process_data.py on failed cases only
    print("\n" + "="*70)
    print("Re-running parameterization for failed cases...")
    print("="*70)

    cmd = [
        "python", "scripts/01_process_data.py",
        "--hariboss_csv", str(temp_csv),
        "--output_dir", "data/processed",
        "--parameterize_rna"
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print("\n✓ Reprocessing completed successfully")
    else:
        print(f"\n✗ Reprocessing failed with return code {result.returncode}")

    # Clean up temp file
    temp_csv.unlink()

    print("\n" + "="*70)
    print("Done! Now you can re-run:")
    print("  python scripts/03_build_dataset.py")
    print("="*70)

if __name__ == "__main__":
    main()
