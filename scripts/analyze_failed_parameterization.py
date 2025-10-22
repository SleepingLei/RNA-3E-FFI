#!/usr/bin/env python3
"""
Analyze why RNA parameterization failed for certain complexes
"""
import json
from pathlib import Path
import pandas as pd

def analyze_processing_results(results_file, amber_dir):
    """
    Analyze the processing results from 01_process_data.py

    Args:
        results_file: Path to processing_results.json
        amber_dir: Path to AMBER directory
    """
    results_file = Path(results_file)
    amber_dir = Path(amber_dir)

    print(f"Reading results from: {results_file}")

    if not results_file.exists():
        print(f"⚠️  Results file not found: {results_file}")
        return

    with open(results_file, 'r') as f:
        results = json.load(f)

    print(f"Found {len(results)} processing results\n")

    # Analyze RNA parameterization results
    rna_success = []
    rna_failed = []
    rna_skipped = []

    for result in results:
        pdb_id = result.get('pdb_id', 'unknown')
        ligand = result.get('ligand', 'unknown')
        model_id = result.get('model_id', 0)
        components = result.get('components', {})

        if 'rna' in components:
            rna_info = components['rna']
            complex_id = f"{pdb_id}_{ligand}_model{model_id}"

            if rna_info.get('success'):
                rna_success.append({
                    'complex_id': complex_id,
                    'atoms': rna_info.get('atoms', 0),
                    'residues': rna_info.get('residues', 0)
                })
            elif rna_info.get('skipped'):
                rna_skipped.append(complex_id)
            else:
                rna_failed.append({
                    'complex_id': complex_id,
                    'atoms': rna_info.get('atoms', 0),
                    'residues': rna_info.get('residues', 0),
                    'errors': result.get('errors', [])
                })

    print("="*80)
    print("RNA PARAMETERIZATION SUMMARY")
    print("="*80)
    print(f"Successful:  {len(rna_success)}")
    print(f"Failed:      {len(rna_failed)}")
    print(f"Skipped:     {len(rna_skipped)}")
    print()

    # Check file existence for failed cases
    if rna_failed:
        print(f"\n{'='*80}")
        print(f"ANALYZING {len(rna_failed)} FAILED RNA PARAMETERIZATIONS")
        print("="*80)

        file_issues = []

        for failed in rna_failed[:50]:  # Check first 50
            complex_id = failed['complex_id']

            # Check what files exist
            prmtop_file = amber_dir / f"{complex_id}_rna.prmtop"
            inpcrd_file = amber_dir / f"{complex_id}_rna.inpcrd"
            pdb_file = amber_dir / f"{complex_id}_rna.pdb"
            cleaned_file = amber_dir / f"{complex_id}_rna_cleaned.pdb"

            prmtop_exists = prmtop_file.exists()
            prmtop_size = prmtop_file.stat().st_size if prmtop_exists else 0

            inpcrd_exists = inpcrd_file.exists()
            inpcrd_size = inpcrd_file.stat().st_size if inpcrd_exists else 0

            issue = {
                'complex_id': complex_id,
                'atoms': failed['atoms'],
                'residues': failed['residues'],
                'prmtop_exists': prmtop_exists,
                'prmtop_size': prmtop_size,
                'inpcrd_exists': inpcrd_exists,
                'inpcrd_size': inpcrd_size,
                'pdb_exists': pdb_file.exists(),
                'cleaned_exists': cleaned_file.exists(),
                'errors': '; '.join(failed.get('errors', []))
            }

            file_issues.append(issue)

            # Print summary for files with issues
            if prmtop_size == 0 or inpcrd_size == 0:
                print(f"\n{complex_id}:")
                print(f"  Atoms: {failed['atoms']}, Residues: {failed['residues']}")
                print(f"  prmtop: {'EXISTS' if prmtop_exists else 'MISSING'}, size: {prmtop_size} bytes")
                print(f"  inpcrd: {'EXISTS' if inpcrd_exists else 'MISSING'}, size: {inpcrd_size} bytes")
                print(f"  PDB: {'EXISTS' if pdb_file.exists() else 'MISSING'}")
                print(f"  Cleaned PDB: {'EXISTS' if cleaned_file.exists() else 'MISSING'}")
                if failed.get('errors'):
                    print(f"  Errors: {failed['errors']}")

        # Save to CSV
        issues_df = pd.DataFrame(file_issues)
        issues_file = amber_dir.parent / "failed_rna_parameterization_analysis.csv"
        issues_df.to_csv(issues_file, index=False)
        print(f"\n\nDetailed analysis saved to: {issues_file}")

        # Statistics
        empty_prmtop = sum(1 for i in file_issues if i['prmtop_size'] == 0)
        empty_inpcrd = sum(1 for i in file_issues if i['inpcrd_size'] == 0)
        both_empty = sum(1 for i in file_issues if i['prmtop_size'] == 0 and i['inpcrd_size'] == 0)

        print(f"\n{'='*80}")
        print("FILE ISSUES STATISTICS")
        print("="*80)
        print(f"Empty prmtop files:      {empty_prmtop}")
        print(f"Empty inpcrd files:      {empty_inpcrd}")
        print(f"Both files empty:        {both_empty}")

def check_tleap_logs(amber_dir):
    """
    Check for tleap log files that might contain error messages

    Args:
        amber_dir: Path to AMBER directory
    """
    amber_dir = Path(amber_dir)

    print(f"\n{'='*80}")
    print("CHECKING FOR TLEAP LOGS")
    print("="*80)

    # Look for any log files or tleap input scripts
    log_files = list(amber_dir.glob("*.log"))
    leap_files = list(amber_dir.glob("*_tleap.in"))

    print(f"Found {len(log_files)} log files")
    print(f"Found {len(leap_files)} tleap input scripts")

    if leap_files:
        print(f"\n⚠️  Warning: Found {len(leap_files)} tleap input scripts that were not cleaned up")
        print("This suggests tleap may have failed for these complexes.")
        print("\nFirst 10 tleap scripts:")
        for f in leap_files[:10]:
            print(f"  - {f.name}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze failed RNA parameterization")
    parser.add_argument("--results_file", type=str, default="data/processing_results.json",
                        help="Path to processing_results.json")
    parser.add_argument("--amber_dir", type=str, default="data/processed/amber",
                        help="Directory containing AMBER files")

    args = parser.parse_args()

    analyze_processing_results(args.results_file, args.amber_dir)
    check_tleap_logs(args.amber_dir)

    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print("="*80)
    print("1. Run the debug_prmtop_files.py script to get detailed file diagnostics")
    print("2. Check the tleap input scripts (*_tleap.in) to see what went wrong")
    print("3. Look for patterns in failed complexes (e.g., modified residues, fragments)")
    print("4. Consider re-running 01_process_data.py with different parameters")
    print()
    print("Commands:")
    print(f"  python scripts/debug_prmtop_files.py --amber_dir {args.amber_dir}")
    print(f"  python scripts/debug_prmtop_files.py --amber_dir {args.amber_dir} --check_specific 7ych 7yci")

if __name__ == "__main__":
    main()
