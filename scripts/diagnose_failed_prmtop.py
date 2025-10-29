#!/usr/bin/env python3
"""
Diagnose why prmtop files are empty for failed cases.

This script analyzes failed graph construction cases where RNA prmtop files are empty (0 bytes).
"""
import sys
from pathlib import Path
import pandas as pd
import subprocess
import MDAnalysis as mda

def analyze_pocket_file(pocket_pdb_path):
    """Analyze pocket PDB file to understand its content."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {pocket_pdb_path.name}")
    print(f"{'='*70}")

    if not pocket_pdb_path.exists():
        print("  ✗ Pocket PDB file not found")
        return None

    # Load with MDAnalysis
    u = mda.Universe(str(pocket_pdb_path))

    print(f"\nBasic Info:")
    print(f"  Total atoms: {len(u.atoms)}")
    print(f"  Total residues: {len(u.residues)}")

    # Separate RNA and ligand
    rna_atoms = u.select_atoms("nucleic")
    ligand_atoms = u.select_atoms("not (nucleic or protein or name HOH or resname HOH WAT TIP3 Na+ Cl- K+ Mg2+)")
    protein_atoms = u.select_atoms("protein")

    print(f"\nComponents:")
    print(f"  RNA atoms: {len(rna_atoms)}")
    print(f"  RNA residues: {len(rna_atoms.residues)}")
    print(f"  Ligand atoms: {len(ligand_atoms)}")
    print(f"  Protein atoms: {len(protein_atoms)}")

    if len(rna_atoms) > 0:
        print(f"\nRNA Residues:")
        for res in rna_atoms.residues:
            print(f"    {res.resname}:{res.resid} ({len(res.atoms)} atoms, chain {res.segid})")

    if len(ligand_atoms) > 0:
        print(f"\nLigand Residues:")
        for res in ligand_atoms.residues:
            print(f"    {res.resname}:{res.resid} ({len(res.atoms)} atoms, chain {res.segid})")

    # Check for terminal atoms that might cause issues
    if len(rna_atoms) > 0:
        print(f"\nRNA Terminal Analysis:")

        # Check first residue
        first_res = rna_atoms.residues[0]
        print(f"  First residue: {first_res.resname}:{first_res.resid}")
        print(f"    Atoms: {', '.join([a.name for a in first_res.atoms])}")

        # Check for 5' phosphate
        has_5p_phosphate = any(a.name in ['P', 'OP1', 'OP2', 'O5\''] for a in first_res.atoms)
        print(f"    Has 5' phosphate: {has_5p_phosphate}")

        # Check last residue
        last_res = rna_atoms.residues[-1]
        print(f"  Last residue: {last_res.resname}:{last_res.resid}")
        print(f"    Atoms: {', '.join([a.name for a in last_res.atoms])}")

        # Check for 3' hydroxyl
        has_3p_hydroxyl = any(a.name == "O3'" for a in last_res.atoms)
        print(f"    Has 3' hydroxyl: {has_3p_hydroxyl}")

        # Check for strange residue names
        unique_resnames = set([res.resname for res in rna_atoms.residues])
        standard_rna = {'A', 'G', 'C', 'U', 'T', 'DA', 'DG', 'DC', 'DT'}
        non_standard = unique_resnames - standard_rna

        if non_standard:
            print(f"\n  ⚠️  Non-standard RNA residues found: {non_standard}")
            for resname in non_standard:
                modified_residues = rna_atoms.select_atoms(f"resname {resname}").residues
                print(f"    {resname}: {len(modified_residues)} residues")
                for res in modified_residues:
                    print(f"      - {res.resname}:{res.resid} ({len(res.atoms)} atoms)")

    return {
        'total_atoms': len(u.atoms),
        'rna_atoms': len(rna_atoms),
        'rna_residues': len(rna_atoms.residues),
        'ligand_atoms': len(ligand_atoms),
        'has_rna': len(rna_atoms) > 0
    }


def check_prmtop_file(prmtop_path):
    """Check prmtop file status."""
    print(f"\nPrmtop File Check:")
    print(f"  Path: {prmtop_path}")

    if not prmtop_path.exists():
        print(f"  ✗ File does not exist")
        return None

    size = prmtop_path.stat().st_size
    print(f"  Size: {size} bytes")

    if size == 0:
        print(f"  ⚠️  File is empty (0 bytes)")
        return 0
    elif size < 100:
        print(f"  ⚠️  File is suspiciously small")
        # Try to read first few lines
        try:
            with open(prmtop_path, 'r') as f:
                lines = f.readlines()[:5]
                print(f"  Content preview:")
                for line in lines:
                    print(f"    {line.rstrip()}")
        except Exception as e:
            print(f"  Error reading file: {e}")
        return size
    else:
        print(f"  ✓ File appears valid")
        return size


def test_rna_parameterization(pocket_pdb_path, output_dir):
    """Test RNA parameterization manually."""
    print(f"\n{'='*70}")
    print(f"Testing RNA Parameterization")
    print(f"{'='*70}")

    # Load pocket
    u = mda.Universe(str(pocket_pdb_path))
    rna_atoms = u.select_atoms("nucleic")

    if len(rna_atoms) == 0:
        print("  ✗ No RNA atoms found in pocket")
        return False

    # Save RNA to temporary PDB
    test_dir = output_dir / "test_parameterization"
    test_dir.mkdir(parents=True, exist_ok=True)

    rna_pdb = test_dir / f"{pocket_pdb_path.stem}_rna_test.pdb"
    rna_atoms.write(str(rna_pdb))
    print(f"  Saved RNA to {rna_pdb.name}")

    # Create tleap script
    tleap_script = test_dir / f"{pocket_pdb_path.stem}_test_tleap.in"
    prmtop_file = test_dir / f"{pocket_pdb_path.stem}_test_rna.prmtop"
    inpcrd_file = test_dir / f"{pocket_pdb_path.stem}_test_rna.inpcrd"

    script_content = f"""source leaprc.RNA.OL3
mol = loadpdb {rna_pdb.name}
set default nocenter on
set default PBRadii mbondi3
check mol

# Try to save even with warnings
saveamberparm mol {prmtop_file.name} {inpcrd_file.name}
quit
"""

    tleap_script.write_text(script_content)
    print(f"  Created tleap script")

    # Run tleap
    print(f"  Running tleap...")
    result = subprocess.run(
        ["tleap", "-f", tleap_script.name],
        capture_output=True,
        text=True,
        cwd=test_dir
    )

    print(f"\n  tleap stdout:")
    print("  " + "\n  ".join(result.stdout.split('\n')[:30]))

    if result.stderr:
        print(f"\n  tleap stderr:")
        print("  " + "\n  ".join(result.stderr.split('\n')[:20]))

    # Check output
    if prmtop_file.exists():
        size = prmtop_file.stat().st_size
        print(f"\n  ✓ prmtop created: {size} bytes")
        if size == 0:
            print(f"  ⚠️  But file is empty!")
            return False
        return True
    else:
        print(f"\n  ✗ prmtop file not created")
        return False


def main():
    # Read failed cases
    failed_csv = Path("data/processed/failed_graph_construction_1.csv")

    if not failed_csv.exists():
        print(f"Error: {failed_csv} not found")
        sys.exit(1)

    # Read CSV
    failed_df = pd.read_csv(failed_csv, header=None, names=['complex_id', 'reason'])

    # Filter only empty prmtop cases
    empty_prmtop_cases = failed_df[failed_df['reason'].str.contains('rna_prmtop_empty', na=False)]

    print(f"Total failed cases: {len(failed_df)}")
    print(f"Empty prmtop cases: {len(empty_prmtop_cases)}")

    # Sample a few cases to analyze
    sample_size = min(5, len(empty_prmtop_cases))
    sample_cases = empty_prmtop_cases.head(sample_size)

    print(f"\nAnalyzing {sample_size} sample cases...")

    for idx, row in sample_cases.iterrows():
        complex_id = row['complex_id']

        # Find pocket file
        pocket_pdb = Path(f"data/processed/pockets/{complex_id}_pocket.pdb")

        # Analyze pocket
        info = analyze_pocket_file(pocket_pdb)

        if info and info['has_rna']:
            # Check prmtop file
            prmtop_path = Path(f"data/processed/amber/{complex_id}_rna.prmtop")
            check_prmtop_file(prmtop_path)

            # Test parameterization
            test_rna_parameterization(pocket_pdb, Path("data/processed"))

        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
