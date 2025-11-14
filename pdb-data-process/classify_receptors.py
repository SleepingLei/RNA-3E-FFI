#!/usr/bin/env python3
"""
Classify and copy receptor files based on their composition (RNA, Protein, or Complex).
This script processes ligand files from processed_ligands_effect_* directories,
finds corresponding receptor files in processed_polymers_fixed, classifies them,
and copies them to the appropriate subdirectories.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import argparse


# Define residue types
RNA_RESIDUES = {
    # Standard RNA/DNA nucleotides
    'A', 'U', 'G', 'C', 'T',
    'DA', 'DT', 'DG', 'DC',
    # Modified nucleotides (common ones)
    'ADE', 'CYT', 'GUA', 'THY', 'URA',
    '+A', '+C', '+G', '+U', '+T',
    'PSU', '5MU', '5MC', '7MG', '2MG', '1MA', 'M2G',
    'OMC', 'OMG', 'OMU',
}

PROTEIN_RESIDUES = {
    # Standard 20 amino acids
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL',
    # Histidine variants
    'HID', 'HIE', 'HIP',
    # Other variants
    'CYX', 'CYM',
    'ASH', 'GLH',
    'LYN',
    # Alternative names
    'MSE',  # Selenomethionine
}


def extract_pdb_id(ligand_filename):
    """
    Extract PDB ID from ligand filename.
    Example: '100D-assembly1_ligands.pdb' -> '100D-assembly1'
    """
    return ligand_filename.replace('_ligands.pdb', '')


def find_receptor_file(pdb_id, receptor_dir):
    """
    Find the corresponding receptor file for a given PDB ID.
    Example: '100D-assembly1' -> '100D-assembly1_polymer_fixed.pdb'
    """
    receptor_file = os.path.join(receptor_dir, f"{pdb_id}_polymer_fixed.pdb")
    if os.path.exists(receptor_file):
        return receptor_file
    return None


def classify_receptor(receptor_file):
    """
    Classify receptor as RNA, Protein, or Complex based on residue types.
    Returns: ('RNA' | 'Protein' | 'Complex', set_of_residues)
    """
    residues = set()

    with open(receptor_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                # Extract residue name (columns 18-20 in PDB format)
                residue = line[17:20].strip()
                residues.add(residue)

    # Check for RNA and protein residues
    has_rna = bool(residues & RNA_RESIDUES)
    has_protein = bool(residues & PROTEIN_RESIDUES)

    if has_rna and has_protein:
        return 'Complex', residues
    elif has_rna:
        return 'RNA', residues
    elif has_protein:
        return 'Protein', residues
    else:
        # Unknown residues - classify based on pattern
        # If contains single letters, likely nucleotides
        single_letter = {r for r in residues if len(r) == 1 or len(r) == 2 and r.startswith('D')}
        if single_letter:
            return 'RNA', residues
        else:
            return 'Unknown', residues


def process_single_file(args):
    """
    Process a single ligand file and its corresponding receptor.
    Designed for multiprocessing.
    """
    ligand_file, receptor_dir, output_base_dir = args

    try:
        # Extract PDB ID
        ligand_filename = os.path.basename(ligand_file)
        pdb_id = extract_pdb_id(ligand_filename)

        # Find receptor file
        receptor_file = find_receptor_file(pdb_id, receptor_dir)

        if not receptor_file:
            return {
                'pdb_id': pdb_id,
                'status': 'receptor_not_found',
                'classification': None,
                'residues': set()
            }

        # Classify receptor
        classification, residues = classify_receptor(receptor_file)

        # Create output directory
        output_dir = os.path.join(output_base_dir, classification)
        os.makedirs(output_dir, exist_ok=True)

        # Copy receptor file
        output_file = os.path.join(output_dir, os.path.basename(receptor_file))
        shutil.copy2(receptor_file, output_file)

        return {
            'pdb_id': pdb_id,
            'status': 'success',
            'classification': classification,
            'residues': residues,
            'receptor_file': receptor_file,
            'output_file': output_file
        }

    except Exception as e:
        return {
            'pdb_id': pdb_id if 'pdb_id' in locals() else 'unknown',
            'status': f'error: {str(e)}',
            'classification': None,
            'residues': set()
        }


def main():
    parser = argparse.ArgumentParser(
        description='Classify and copy receptor files based on composition'
    )
    parser.add_argument('--ligand-dir', default='processed_ligands_effect_1',
                        help='Directory containing ligand PDB files (default: processed_ligands_effect_1)')
    parser.add_argument('--receptor-dir', default='processed_polymers_fixed',
                        help='Directory containing receptor PDB files (default: processed_polymers_fixed)')
    parser.add_argument('--output-dir', default='effect_receptor',
                        help='Output base directory (default: effect_receptor)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count)')
    parser.add_argument('--all-ligand-dirs', action='store_true',
                        help='Process all processed_ligands_effect_* directories')

    args = parser.parse_args()

    # Determine which ligand directories to process
    if args.all_ligand_dirs:
        # Find all processed_ligands_effect_* directories
        ligand_dirs = sorted([
            d for d in os.listdir('.')
            if os.path.isdir(d) and d.startswith('processed_ligands_effect_')
        ])
        if not ligand_dirs:
            print("No processed_ligands_effect_* directories found!")
            return
        print(f"Found {len(ligand_dirs)} ligand directories to process:")
        for d in ligand_dirs:
            print(f"  - {d}")
        print()
    else:
        ligand_dirs = [args.ligand_dir]

    receptor_dir = args.receptor_dir
    output_base_dir = args.output_dir

    # Verify receptor directory exists
    if not os.path.exists(receptor_dir):
        print(f"Error: Receptor directory '{receptor_dir}' not found!")
        return

    # Collect all ligand files
    all_ligand_files = []
    for ligand_dir in ligand_dirs:
        if not os.path.exists(ligand_dir):
            print(f"Warning: Ligand directory '{ligand_dir}' not found, skipping...")
            continue

        ligand_files = [
            os.path.join(ligand_dir, f)
            for f in os.listdir(ligand_dir)
            if f.endswith('.pdb')
        ]
        all_ligand_files.extend(ligand_files)

    if not all_ligand_files:
        print("No ligand files found!")
        return

    print(f"Found {len(all_ligand_files)} ligand files to process")
    print(f"Receptor directory: {receptor_dir}")
    print(f"Output directory: {output_base_dir}")
    print()

    # Prepare arguments for multiprocessing
    process_args = [
        (ligand_file, receptor_dir, output_base_dir)
        for ligand_file in all_ligand_files
    ]

    # Process files
    num_workers = args.workers or cpu_count()
    print(f"Processing with {num_workers} worker processes...\n")

    with Pool(num_workers) as pool:
        results = pool.map(process_single_file, process_args)

    # Collect statistics
    stats = defaultdict(list)
    success_count = 0
    not_found_count = 0
    error_count = 0

    for result in results:
        if result['status'] == 'success':
            success_count += 1
            stats[result['classification']].append(result)
        elif result['status'] == 'receptor_not_found':
            not_found_count += 1
            print(f"Warning: Receptor not found for {result['pdb_id']}")
        else:
            error_count += 1
            print(f"Error processing {result['pdb_id']}: {result['status']}")

    # Print statistics
    print("\n" + "="*80)
    print("CLASSIFICATION COMPLETE")
    print("="*80)
    print(f"\nTotal ligand files processed: {len(all_ligand_files)}")
    print(f"Successfully classified: {success_count}")
    print(f"Receptor not found: {not_found_count}")
    print(f"Errors: {error_count}")

    print("\n" + "-"*80)
    print("CLASSIFICATION SUMMARY")
    print("-"*80)

    for classification in ['RNA', 'Protein', 'Complex', 'Unknown']:
        if classification in stats:
            count = len(stats[classification])
            print(f"\n{classification}: {count} files")
            print(f"  Output directory: {output_base_dir}/{classification}/")

            # Show first few examples with residue info
            for i, result in enumerate(stats[classification][:5]):
                residue_list = sorted(result['residues'])
                residue_str = ', '.join(residue_list[:10])
                if len(residue_list) > 10:
                    residue_str += f", ... ({len(residue_list)} total)"
                print(f"    - {result['pdb_id']}: {residue_str}")

            if count > 5:
                print(f"    ... and {count - 5} more files")

    # Summary table
    print("\n" + "-"*80)
    print("SUMMARY TABLE")
    print("-"*80)
    print(f"{'Classification':<20} {'Count':<15} {'Percentage':<15}")
    print("-"*80)

    total_classified = sum(len(stats[c]) for c in stats)
    for classification in ['RNA', 'Protein', 'Complex', 'Unknown']:
        if classification in stats:
            count = len(stats[classification])
            percentage = (count / total_classified * 100) if total_classified > 0 else 0
            print(f"{classification:<20} {count:<15} {percentage:>6.2f}%")

    print("-"*80)
    print(f"{'Total':<20} {total_classified:<15} 100.00%")
    print("-"*80)

    # Save detailed report
    report_file = os.path.join(output_base_dir, 'classification_report.txt')
    with open(report_file, 'w') as f:
        f.write("RECEPTOR CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total ligand files processed: {len(all_ligand_files)}\n")
        f.write(f"Successfully classified: {success_count}\n")
        f.write(f"Receptor not found: {not_found_count}\n")
        f.write(f"Errors: {error_count}\n\n")

        f.write("DETAILED RESULTS BY CLASSIFICATION\n")
        f.write("-"*80 + "\n\n")

        for classification in ['RNA', 'Protein', 'Complex', 'Unknown']:
            if classification in stats:
                f.write(f"\n{classification}: {len(stats[classification])} files\n")
                f.write("-"*40 + "\n")
                for result in stats[classification]:
                    residue_list = sorted(result['residues'])
                    residue_str = ', '.join(residue_list)
                    f.write(f"{result['pdb_id']:<30} {residue_str}\n")

        # Summary statistics
        f.write("\n\nSUMMARY STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Classification':<20} {'Count':<15} {'Percentage':<15}\n")
        f.write("-"*80 + "\n")

        for classification in ['RNA', 'Protein', 'Complex', 'Unknown']:
            if classification in stats:
                count = len(stats[classification])
                percentage = (count / total_classified * 100) if total_classified > 0 else 0
                f.write(f"{classification:<20} {count:<15} {percentage:>6.2f}%\n")

        f.write("-"*80 + "\n")
        f.write(f"{'Total':<20} {total_classified:<15} 100.00%\n")

    print(f"\nDetailed report saved to: {report_file}")
    print("\nDone!\n")


if __name__ == '__main__':
    main()
