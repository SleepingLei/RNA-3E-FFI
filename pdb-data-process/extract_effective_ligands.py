#!/usr/bin/env python3
"""
Extract effective small molecules from PDB ligand files.
Excludes cofactors and other molecules listed in exclude_molecules.txt.
Organizes output by number of effective molecules per file.
"""

import os
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import argparse


def load_exclude_molecules(exclude_file):
    """Load the list of molecules to exclude from the exclude file."""
    exclude_set = set()

    if not os.path.exists(exclude_file):
        print(f"Warning: Exclude file {exclude_file} not found!")
        return exclude_set

    with open(exclude_file, 'r') as f:
        content = f.read()
        # Parse Python set notation or simple list
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            # Extract molecule name from quotes
            if "'" in line or '"' in line:
                parts = line.split("'")
                if len(parts) < 2:
                    parts = line.split('"')
                if len(parts) >= 2:
                    mol_name = parts[1].strip()
                    if mol_name:
                        exclude_set.add(mol_name)

    return exclude_set


def extract_effective_molecules(pdb_file, exclude_molecules):
    """
    Extract effective small molecules from a PDB file.
    Returns a dictionary with residue names as keys and their HETATM lines as values.
    """
    effective_molecules = defaultdict(list)
    current_residue = None
    current_lines = []

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('HETATM'):
                # Extract residue name (columns 18-20 in PDB format)
                residue_name = line[17:20].strip()

                # Check if this is a new residue
                if current_residue is None:
                    current_residue = residue_name
                    current_lines = [line]
                elif residue_name == current_residue:
                    current_lines.append(line)
                else:
                    # Save the previous residue if it's not excluded
                    if current_residue not in exclude_molecules:
                        effective_molecules[current_residue].extend(current_lines)
                    # Start new residue
                    current_residue = residue_name
                    current_lines = [line]

            elif line.startswith('TER') and current_residue:
                # End of current residue
                if current_residue not in exclude_molecules:
                    effective_molecules[current_residue].extend(current_lines)
                    effective_molecules[current_residue].append(line)
                current_residue = None
                current_lines = []

    # Handle last residue if file doesn't end with TER
    if current_residue and current_lines:
        if current_residue not in exclude_molecules:
            effective_molecules[current_residue].extend(current_lines)

    return effective_molecules


def process_single_file(args):
    """Process a single PDB file. Designed for multiprocessing."""
    pdb_file, exclude_molecules, source_dir, base_output_dir = args

    try:
        # Extract effective molecules
        effective_molecules = extract_effective_molecules(pdb_file, exclude_molecules)

        # Count unique effective molecules
        num_effective = len(effective_molecules)

        if num_effective == 0:
            return {
                'filename': os.path.basename(pdb_file),
                'num_effective': 0,
                'molecules': [],
                'status': 'no_effective_molecules'
            }

        # Create output directory based on count
        output_dir = os.path.join(base_output_dir, f'processed_ligands_effect_{num_effective}')
        os.makedirs(output_dir, exist_ok=True)

        # Write the effective molecules to a new PDB file
        output_file = os.path.join(output_dir, os.path.basename(pdb_file))

        with open(output_file, 'w') as f:
            # Write header
            f.write(f"REMARK   1 EFFECTIVE LIGANDS EXTRACTED\n")
            f.write(f"REMARK   1 SOURCE FILE: {os.path.basename(pdb_file)}\n")
            f.write(f"REMARK   1 NUMBER OF EFFECTIVE MOLECULES: {num_effective}\n")
            f.write(f"REMARK   1 MOLECULES: {', '.join(effective_molecules.keys())}\n")

            # Write all effective molecule lines
            for residue_name, lines in effective_molecules.items():
                for line in lines:
                    f.write(line)

            # Write END
            f.write("END\n")

        return {
            'filename': os.path.basename(pdb_file),
            'num_effective': num_effective,
            'molecules': list(effective_molecules.keys()),
            'status': 'success',
            'output_file': output_file
        }

    except Exception as e:
        return {
            'filename': os.path.basename(pdb_file),
            'num_effective': 0,
            'molecules': [],
            'status': f'error: {str(e)}'
        }


def main():
    parser = argparse.ArgumentParser(description='Extract effective ligands from PDB files')
    parser.add_argument('--input-dir', default='processed_ligands',
                        help='Input directory containing ligand PDB files')
    parser.add_argument('--exclude-file', default='processed_ligands/exclude_molecules.txt',
                        help='File containing molecules to exclude')
    parser.add_argument('--output-base', default='.',
                        help='Base directory for output folders')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count)')

    args = parser.parse_args()

    # Setup paths
    input_dir = args.input_dir
    exclude_file = args.exclude_file
    output_base = args.output_base

    # Load exclude molecules
    print(f"Loading exclude molecules from {exclude_file}...")
    exclude_molecules = load_exclude_molecules(exclude_file)
    print(f"Loaded {len(exclude_molecules)} molecules to exclude: {sorted(exclude_molecules)}\n")

    # Get all PDB files
    pdb_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith('.pdb')
    ])

    print(f"Found {len(pdb_files)} PDB files to process\n")

    if len(pdb_files) == 0:
        print("No PDB files found!")
        return

    # Prepare arguments for multiprocessing
    process_args = [
        (pdb_file, exclude_molecules, input_dir, output_base)
        for pdb_file in pdb_files
    ]

    # Process files using multiprocessing
    num_workers = args.workers or cpu_count()
    print(f"Processing with {num_workers} worker processes...\n")

    with Pool(num_workers) as pool:
        results = pool.map(process_single_file, process_args)

    # Collect statistics
    stats = defaultdict(list)
    success_count = 0
    error_count = 0
    no_effective_count = 0

    for result in results:
        if result['status'] == 'success':
            success_count += 1
            stats[result['num_effective']].append(result)
        elif result['status'] == 'no_effective_molecules':
            no_effective_count += 1
            stats[0].append(result)
        else:
            error_count += 1
            print(f"Error processing {result['filename']}: {result['status']}")

    # Print statistics
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"\nTotal files processed: {len(pdb_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Files with no effective molecules: {no_effective_count}")
    print(f"Errors: {error_count}")

    print("\n" + "-"*80)
    print("DISTRIBUTION BY NUMBER OF EFFECTIVE MOLECULES")
    print("-"*80)

    for num_molecules in sorted(stats.keys()):
        if num_molecules > 0:
            print(f"\n{num_molecules} effective molecule(s): {len(stats[num_molecules])} files")
            print(f"  Output directory: processed_ligands_effect_{num_molecules}/")

            # Show first few examples
            for i, result in enumerate(stats[num_molecules][:5]):
                molecules_str = ', '.join(result['molecules'])
                print(f"    - {result['filename']}: {molecules_str}")

            if len(stats[num_molecules]) > 5:
                print(f"    ... and {len(stats[num_molecules]) - 5} more files")

    if 0 in stats:
        print(f"\n0 effective molecules: {len(stats[0])} files")
        print(f"  (These files contain only excluded molecules)")

    # Summary table
    print("\n" + "-"*80)
    print("SUMMARY TABLE")
    print("-"*80)
    print(f"{'Effective Molecules':<25} {'Number of Files':<20}")
    print("-"*80)
    for num_molecules in sorted(stats.keys()):
        if num_molecules > 0:
            print(f"{num_molecules:<25} {len(stats[num_molecules]):<20}")
    if 0 in stats:
        print(f"{'0 (no effective)':<25} {len(stats[0]):<20}")
    print("-"*80)

    # Save detailed report
    report_file = os.path.join(output_base, 'ligand_extraction_report.txt')
    with open(report_file, 'w') as f:
        f.write("LIGAND EXTRACTION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total files processed: {len(pdb_files)}\n")
        f.write(f"Successfully processed: {success_count}\n")
        f.write(f"Files with no effective molecules: {no_effective_count}\n")
        f.write(f"Errors: {error_count}\n\n")

        f.write("DETAILED RESULTS\n")
        f.write("-"*80 + "\n\n")

        for num_molecules in sorted(stats.keys()):
            f.write(f"\n{num_molecules} effective molecule(s): {len(stats[num_molecules])} files\n")
            f.write("-"*40 + "\n")
            for result in stats[num_molecules]:
                molecules_str = ', '.join(result['molecules']) if result['molecules'] else 'None'
                f.write(f"{result['filename']:<40} {molecules_str}\n")

    print(f"\nDetailed report saved to: {report_file}")
    print("\nDone!\n")


if __name__ == '__main__':
    main()
