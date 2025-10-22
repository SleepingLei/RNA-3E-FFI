#!/usr/bin/env python3
"""
Demo script showcasing new ligand and modified RNA parameterization features
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.process_data_01 import process_complex_v2

def demo_single_complex():
    """
    Demo: Process a single complex with ligand and potentially modified RNA
    """
    print("\n" + "="*70)
    print("DEMO: New Parameterization Features")
    print("="*70)

    # Example: Process one complex from HARIBOSS
    # You can change these parameters
    pdb_id = "1aq3"          # Replace with actual PDB ID from HARIBOSS
    ligand_name = "ARG"       # Replace with actual ligand name
    pocket_cutoff = 5.0       # Angstroms

    # Paths
    hariboss_dir = Path("hariboss")
    output_dir = Path("data/demo")

    print(f"\nProcessing complex: {pdb_id}")
    print(f"Target ligand: {ligand_name}")
    print(f"Pocket cutoff: {pocket_cutoff} Å")

    # Check if files exist
    cif_file = Path("data/raw/mmCIF") / f"{pdb_id}.cif"
    if not cif_file.exists():
        cif_file = hariboss_dir / "mmCIF" / f"{pdb_id}.cif"

    if not cif_file.exists():
        print(f"\n⚠️  CIF file not found: {pdb_id}.cif")
        print(f"   Please ensure the file exists in either:")
        print(f"   - data/raw/mmCIF/{pdb_id}.cif")
        print(f"   - {hariboss_dir}/mmCIF/{pdb_id}.cif")
        return

    # Process
    result = process_complex_v2(
        pdb_id=pdb_id,
        ligand_name=ligand_name,
        hariboss_dir=hariboss_dir,
        output_dir=output_dir,
        pocket_cutoff=pocket_cutoff
    )

    # Show results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\nPDB ID: {result['pdb_id']}")
    print(f"Ligand: {result['ligand']}")
    print(f"Overall success: {result['success']}")

    if result['components']:
        print("\nComponent parameterization:")
        for comp_type, comp_data in result['components'].items():
            print(f"\n  {comp_type.upper()}:")
            print(f"    Success: {comp_data['success']}")
            print(f"    Atoms: {comp_data['atoms']}")
            if 'residues' in comp_data:
                print(f"    Residues: {comp_data['residues']}")
            if comp_data.get('prmtop'):
                print(f"    Topology: {comp_data['prmtop']}")
                print(f"    Coords: {comp_data['inpcrd']}")

    if result['errors']:
        print("\n  Errors:")
        for error in result['errors']:
            print(f"    - {error}")

    # Show generated files
    print("\n" + "="*70)
    print("GENERATED FILES")
    print("="*70)

    amber_dir = output_dir / "amber"
    if amber_dir.exists():
        print(f"\nAmber parameter files in: {amber_dir}")
        for f in sorted(amber_dir.glob(f"{pdb_id}_{ligand_name}*")):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name:50s} ({size_kb:6.1f} KB)")

    pocket_dir = output_dir / "pockets"
    if pocket_dir.exists():
        print(f"\nPocket files in: {pocket_dir}")
        for f in sorted(pocket_dir.glob(f"{pdb_id}_{ligand_name}*")):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name:50s} ({size_kb:6.1f} KB)")


def demo_workflow_summary():
    """
    Print a summary of the new workflow
    """
    print("\n" + "="*70)
    print("NEW FEATURE WORKFLOW SUMMARY")
    print("="*70)

    print("""
The new parameterization pipeline handles:

1. STANDARD RNA
   - Force field: RNA.OL3
   - Tool: tleap with leaprc.RNA.OL3
   - Output: *_rna.prmtop/inpcrd

2. MODIFIED RNA (PSU, 5MU, 7MG, etc.)
   - Force field: GAFF2
   - Tools: antechamber → parmchk2 → tleap
   - Strategy: Each residue parameterized separately, then combined
   - Output: *_modified_rna.prmtop/inpcrd

3. LIGANDS (small molecules)
   - Force field: GAFF2
   - Tools: antechamber → parmchk2 → tleap
   - Charges: AM1-BCC (default)
   - Output: *_ligand.prmtop/inpcrd

4. PROTEINS (if present)
   - Force field: ff14SB
   - Tool: tleap with leaprc.protein.ff14SB
   - Output: *_protein.prmtop/inpcrd

Key advantages:
✓ Fully automated parameterization
✓ Handles modified RNA residues
✓ Separate topology files for each component
✓ Ready for MD simulation or GNN training
✓ Robust error handling and logging
    """)

    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Demo script for new parameterization features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show workflow summary
  python scripts/demo_new_features.py --summary

  # Process a demo complex
  python scripts/demo_new_features.py --demo --pdb 1aq3 --ligand ARG

  # Process with custom cutoff
  python scripts/demo_new_features.py --demo --pdb 1aq3 --ligand ARG --cutoff 8.0
        """
    )

    parser.add_argument("--summary", action="store_true",
                       help="Show workflow summary")
    parser.add_argument("--demo", action="store_true",
                       help="Run demo on a single complex")
    parser.add_argument("--pdb", type=str, default="1aq3",
                       help="PDB ID for demo (default: 1aq3)")
    parser.add_argument("--ligand", type=str, default="ARG",
                       help="Ligand name for demo (default: ARG)")
    parser.add_argument("--cutoff", type=float, default=5.0,
                       help="Pocket cutoff in Angstroms (default: 5.0)")

    args = parser.parse_args()

    # If no arguments, show help
    if not args.summary and not args.demo:
        parser.print_help()
        demo_workflow_summary()
        return

    # Show summary
    if args.summary:
        demo_workflow_summary()

    # Run demo
    if args.demo:
        # Update globals for demo function
        global pdb_id, ligand_name, pocket_cutoff
        pdb_id = args.pdb
        ligand_name = args.ligand
        pocket_cutoff = args.cutoff

        demo_single_complex()


if __name__ == "__main__":
    main()
