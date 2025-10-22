#!/usr/bin/env python3
"""
Diagnostic script to identify issues with AMBER prmtop files
"""
import os
import sys
from pathlib import Path
import pandas as pd

def check_prmtop_files(amber_dir):
    """
    Check all prmtop files for common issues

    Args:
        amber_dir: Directory containing AMBER files
    """
    amber_dir = Path(amber_dir)

    print(f"Checking AMBER files in: {amber_dir}\n")
    print("="*80)

    # Find all prmtop files
    prmtop_files = sorted(amber_dir.glob("*.prmtop"))

    if not prmtop_files:
        print("⚠️  No .prmtop files found!")
        return

    print(f"Found {len(prmtop_files)} prmtop files\n")

    # Statistics
    stats = {
        'total': len(prmtop_files),
        'empty': 0,
        'small': 0,  # < 1KB
        'valid': 0,
        'missing_inpcrd': 0,
        'missing_pdb': 0
    }

    issues = []

    for prmtop_file in prmtop_files:
        size = prmtop_file.stat().st_size

        # Check if empty
        if size == 0:
            stats['empty'] += 1
            issues.append({
                'file': prmtop_file.name,
                'issue': 'EMPTY',
                'size': 0,
                'details': 'File is 0 bytes'
            })
            continue

        # Check if suspiciously small
        if size < 1024:
            stats['small'] += 1
            issues.append({
                'file': prmtop_file.name,
                'issue': 'TOO_SMALL',
                'size': size,
                'details': f'File is only {size} bytes'
            })

        # Check for corresponding inpcrd file
        inpcrd_file = prmtop_file.parent / prmtop_file.name.replace('.prmtop', '.inpcrd')
        if not inpcrd_file.exists():
            stats['missing_inpcrd'] += 1
            issues.append({
                'file': prmtop_file.name,
                'issue': 'MISSING_INPCRD',
                'size': size,
                'details': f'No matching {inpcrd_file.name}'
            })

        # Check for corresponding PDB file
        pdb_file = prmtop_file.parent / prmtop_file.name.replace('.prmtop', '.pdb')
        if not pdb_file.exists():
            stats['missing_pdb'] += 1
            issues.append({
                'file': prmtop_file.name,
                'issue': 'MISSING_PDB',
                'size': size,
                'details': f'No matching {pdb_file.name}'
            })

        # Check file header (first few bytes)
        try:
            with open(prmtop_file, 'r') as f:
                first_line = f.readline().strip()
                if not first_line.startswith('%VERSION'):
                    issues.append({
                        'file': prmtop_file.name,
                        'issue': 'INVALID_FORMAT',
                        'size': size,
                        'details': f'Does not start with %VERSION, starts with: {first_line[:50]}'
                    })
                else:
                    stats['valid'] += 1
        except Exception as e:
            issues.append({
                'file': prmtop_file.name,
                'issue': 'READ_ERROR',
                'size': size,
                'details': str(e)
            })

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total prmtop files:        {stats['total']}")
    print(f"Valid files:               {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
    print(f"Empty files (0 bytes):     {stats['empty']}")
    print(f"Small files (< 1KB):       {stats['small']}")
    print(f"Missing .inpcrd:           {stats['missing_inpcrd']}")
    print(f"Missing .pdb:              {stats['missing_pdb']}")

    # Print issues
    if issues:
        print(f"\n{'='*80}")
        print(f"ISSUES FOUND: {len(issues)}")
        print("="*80)

        # Save to CSV
        issues_df = pd.DataFrame(issues)
        issues_file = amber_dir.parent / "prmtop_issues.csv"
        issues_df.to_csv(issues_file, index=False)
        print(f"\nDetailed issues saved to: {issues_file}")

        # Print first 20 issues
        print("\nFirst 20 issues:")
        for i, issue in enumerate(issues[:20], 1):
            print(f"\n{i}. {issue['file']}")
            print(f"   Issue: {issue['issue']}")
            print(f"   Size: {issue['size']} bytes")
            print(f"   Details: {issue['details']}")

        if len(issues) > 20:
            print(f"\n... and {len(issues) - 20} more issues (see CSV file)")
    else:
        print(f"\n✓ No issues found! All prmtop files appear valid.")

    print("\n" + "="*80)

def check_specific_files(amber_dir, pdb_ids):
    """
    Check specific problematic files mentioned in error

    Args:
        amber_dir: Directory containing AMBER files
        pdb_ids: List of PDB IDs to check
    """
    amber_dir = Path(amber_dir)

    print(f"\n{'='*80}")
    print("DETAILED CHECK OF SPECIFIC FILES")
    print("="*80)

    for pdb_id in pdb_ids:
        print(f"\n{pdb_id}:")
        print("-" * 40)

        # Find all files for this PDB ID
        pattern = f"{pdb_id}_*"
        files = sorted(amber_dir.glob(pattern))

        if not files:
            print(f"  ⚠️  No files found for {pdb_id}")
            continue

        print(f"  Found {len(files)} files:")
        for f in files:
            size = f.stat().st_size
            size_str = f"{size:,} bytes" if size > 0 else "EMPTY"
            print(f"    - {f.name:<50} {size_str}")

            # For prmtop files, check content
            if f.suffix == '.prmtop' and size > 0:
                try:
                    with open(f, 'r') as file:
                        first_lines = [file.readline().strip() for _ in range(3)]
                        print(f"      First line: {first_lines[0][:60]}...")
                except Exception as e:
                    print(f"      Error reading: {e}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose AMBER prmtop file issues")
    parser.add_argument("--amber_dir", type=str, default="data/processed/amber",
                        help="Directory containing AMBER files")
    parser.add_argument("--check_specific", type=str, nargs="+",
                        help="Check specific PDB IDs (e.g., 7ych 7yci)")

    args = parser.parse_args()

    # Run general check
    check_prmtop_files(args.amber_dir)

    # Check specific files if requested
    if args.check_specific:
        check_specific_files(args.amber_dir, args.check_specific)

if __name__ == "__main__":
    main()
