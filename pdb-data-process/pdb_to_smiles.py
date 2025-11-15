#!/usr/bin/env python3
"""
Convert PDB files to SMILES with pH correction using OpenBabel.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from multiprocessing import Pool, cpu_count
import argparse
import csv


def extract_molecule_name(pdb_file):
    """Extract molecule name from PDB file (residue name)."""
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('HETATM') or line.startswith('ATOM'):
                # Extract residue name (columns 18-20 in PDB format)
                residue_name = line[17:20].strip()
                if residue_name:
                    return residue_name
            elif line.startswith('REMARK') and 'MOLECULES:' in line:
                # Try to extract from REMARK
                parts = line.split('MOLECULES:')
                if len(parts) > 1:
                    mol_names = parts[1].strip().split(',')
                    if mol_names:
                        return mol_names[0].strip()
    return 'UNK'


def pdb_to_smiles(pdb_file, ph=7.4):
    """
    Convert PDB file to SMILES using OpenBabel.
    Returns (smiles, corrected_smiles)
    """
    try:
        # Convert to SMILES without pH correction
        result = subprocess.run(
            ['obabel', pdb_file, '-osmi'],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return None, None

        # Parse SMILES (first column)
        smiles = result.stdout.strip().split()[0] if result.stdout.strip() else None

        # Convert to SMILES with pH correction
        result_corrected = subprocess.run(
            ['obabel', pdb_file, '-osmi', '-p', str(ph)],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result_corrected.returncode != 0:
            corrected_smiles = smiles  # Fallback to uncorrected
        else:
            corrected_smiles = result_corrected.stdout.strip().split()[0] if result_corrected.stdout.strip() else smiles

        return smiles, corrected_smiles

    except subprocess.TimeoutExpired:
        print(f"Timeout processing {pdb_file}")
        return None, None
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        return None, None


def process_single_file(args):
    """Process a single PDB file. Designed for multiprocessing."""
    pdb_file, ph = args

    try:
        filename = os.path.basename(pdb_file)
        molecule_name = extract_molecule_name(pdb_file)
        smiles, corrected_smiles = pdb_to_smiles(pdb_file, ph)

        return {
            'filename': filename,
            'molecule_name': molecule_name,
            'smiles': smiles if smiles else 'N/A',
            'corrected_smiles': corrected_smiles if corrected_smiles else 'N/A',
            'status': 'success' if smiles else 'failed'
        }

    except Exception as e:
        return {
            'filename': os.path.basename(pdb_file),
            'molecule_name': 'ERROR',
            'smiles': 'N/A',
            'corrected_smiles': 'N/A',
            'status': f'error: {str(e)}'
        }


def main():
    parser = argparse.ArgumentParser(
        description='Convert PDB files to SMILES with pH correction'
    )
    parser.add_argument('--input-dir', default='extracted_ligands/processed_ligands_effect_1',
                        help='Input directory containing PDB files')
    parser.add_argument('--output-csv', default='ligands_smiles.csv',
                        help='Output CSV file')
    parser.add_argument('--ph', type=float, default=7.4,
                        help='pH value for correction (default: 7.4)')
    parser.add_argument('--workers', type=int, default=32,
                        help='Number of worker processes (default: CPU count)')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_csv = args.output_csv
    ph = args.ph

    # Get all PDB files
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found!")
        return

    pdb_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith('.pdb')
    ])

    if not pdb_files:
        print(f"No PDB files found in {input_dir}")
        return

    print(f"Found {len(pdb_files)} PDB files to process")
    print(f"pH correction: {ph}")
    print(f"Output file: {output_csv}\n")

    # Prepare arguments for multiprocessing
    process_args = [(pdb_file, ph) for pdb_file in pdb_files]

    # Process files
    num_workers = args.workers or cpu_count()
    print(f"Processing with {num_workers} worker processes...\n")

    with Pool(num_workers) as pool:
        results = pool.map(process_single_file, process_args)

    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'molecule_name', 'smiles', 'corrected_smiles']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow({
                'filename': result['filename'],
                'molecule_name': result['molecule_name'],
                'smiles': result['smiles'],
                'corrected_smiles': result['corrected_smiles']
            })

    # Print statistics
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = len(results) - success_count

    print("\n" + "="*80)
    print("CONVERSION COMPLETE")
    print("="*80)
    print(f"\nTotal files processed: {len(results)}")
    print(f"Successfully converted: {success_count}")
    print(f"Failed: {failed_count}")

    if failed_count > 0:
        print("\nFailed files:")
        for result in results:
            if result['status'] != 'success':
                print(f"  - {result['filename']}: {result['status']}")

    print(f"\nResults saved to: {output_csv}")
    print("\nDone!\n")


if __name__ == '__main__':
    main()
