#!/usr/bin/env python3
"""
Enhanced RNA-ligand complex processing with separate parameterization
Version 2.0 - Implements divide-and-conquer strategy
"""

import argparse
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import MDAnalysis as mda
from Bio.PDB import MMCIFParser, PDBIO
from tqdm import tqdm
import json

# Residue type definitions
RNA_RESIDUES = ['A', 'C', 'G', 'U', 'A3', 'A5', 'C3', 'C5', 'G3', 'G5', 'U3', 'U5',
                'DA', 'DC', 'DG', 'DT', 'DA3', 'DA5', 'DC3', 'DC5', 'DG3', 'DG5', 'DT3', 'DT5']

MODIFIED_RNA = ['PSU', '5MU', '5MC', '1MA', '7MG', 'M2G', 'OMC', 'OMG', 'H2U',
                '2MG', 'M7G', 'OMU', 'YYG', 'YG', '6MZ', 'IU', 'I']

PROTEIN_RESIDUES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
                    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
                    'TYR', 'VAL', 'HID', 'HIE', 'HIP', 'CYX']

SOLVENT = ['HOH', 'WAT', 'TIP3', 'TIP', 'SOL']
IONS = ['NA', 'CL', 'MG', 'K', 'CA', 'ZN', 'MN', 'FE']

# Common cofactors/ligands that should be treated as ligands, not protein residues
COMMON_LIGANDS = ['ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'GMP', 'NAD', 'FAD', 'SAM',
                  'SAH', 'COA', 'HEM', 'NDP', 'NTP']


class MoleculeClassifier:
    """Classify and separate molecules in a structure"""

    def __init__(self, universe: mda.Universe, target_ligand: str):
        self.u = universe
        self.target_ligand = target_ligand

    def classify_molecules(self) -> Dict[str, mda.AtomGroup]:
        """
        Classify all molecules into categories
        Returns dict with keys: 'rna', 'protein', 'ligand', 'water', 'ions', 'unknown'
        """
        all_atoms = self.u.atoms

        # Initialize categories with empty selections
        categories = {
            'rna': self.u.select_atoms('resname NONE'),
            'modified_rna': self.u.select_atoms('resname NONE'),
            'protein': self.u.select_atoms('resname NONE'),
            'target_ligand': self.u.select_atoms('resname NONE'),
            'other_ligands': self.u.select_atoms('resname NONE'),
            'water': self.u.select_atoms('resname NONE'),
            'ions': self.u.select_atoms('resname NONE'),
            'unknown': self.u.select_atoms('resname NONE')
        }

        # Classify residues
        for residue in self.u.residues:
            resname = residue.resname.strip()
            atoms = residue.atoms

            # Target ligand (highest priority)
            if resname == self.target_ligand:
                categories['target_ligand'] = categories['target_ligand'] + atoms

            # Standard RNA
            elif resname in RNA_RESIDUES:
                categories['rna'] = categories['rna'] + atoms

            # Modified RNA
            elif resname in MODIFIED_RNA:
                categories['modified_rna'] = categories['modified_rna'] + atoms

            # Protein
            elif resname in PROTEIN_RESIDUES:
                categories['protein'] = categories['protein'] + atoms

            # Water
            elif resname in SOLVENT:
                categories['water'] = categories['water'] + atoms

            # Ions
            elif resname in IONS:
                categories['ions'] = categories['ions'] + atoms

            # Other ligands (nucleotides, cofactors, etc.)
            elif resname in COMMON_LIGANDS or len(atoms) < 50:  # Small molecules
                categories['other_ligands'] = categories['other_ligands'] + atoms

            # Unknown
            else:
                categories['unknown'] = categories['unknown'] + atoms

        # Print summary
        print(f"\n{'='*70}")
        print(f"Molecule Classification")
        print(f"{'='*70}")
        for cat, atoms in categories.items():
            if len(atoms) > 0:
                residues = len(atoms.residues)
                print(f"  {cat:20s}: {len(atoms):4d} atoms, {residues:3d} residues")
        print(f"{'='*70}\n")

        return categories


def define_pocket_by_residues(universe: mda.Universe, ligand_resname: str,
                              pocket_cutoff: float) -> Dict[str, mda.AtomGroup]:
    """
    Define pocket using complete residues (not individual atoms)
    Returns separated components of the pocket
    """
    print(f"\nDefining pocket with {pocket_cutoff}Å cutoff around {ligand_resname}...")

    # Get ligand
    ligand = universe.select_atoms(f"resname {ligand_resname}")
    if len(ligand) == 0:
        print(f"⚠️  Warning: No ligand found with resname {ligand_resname}")
        return {}

    print(f"Ligand: {len(ligand)} atoms")

    # Classify all molecules
    classifier = MoleculeClassifier(universe, ligand_resname)
    all_components = classifier.classify_molecules()

    # For each component type, select residues within cutoff
    pocket_components = {}

    # RNA pocket
    if len(all_components['rna']) > 0:
        rna_atoms_in_range = all_components['rna'].select_atoms(
            f"around {pocket_cutoff} global resname {ligand_resname}"
        )
        if len(rna_atoms_in_range) > 0:
            # Get complete residues
            residues_to_include = rna_atoms_in_range.residues
            resindices = [res.resindex for res in residues_to_include]
            pocket_components['rna'] = universe.select_atoms(
                f"resindex {' '.join(map(str, resindices))}"
            )
            print(f"RNA: {len(rna_atoms_in_range)} atoms in {len(residues_to_include)} residues → "
                  f"{len(pocket_components['rna'])} atoms (complete residues)")

    # Modified RNA pocket
    if len(all_components['modified_rna']) > 0:
        mod_rna_in_range = all_components['modified_rna'].select_atoms(
            f"around {pocket_cutoff} global resname {ligand_resname}"
        )
        if len(mod_rna_in_range) > 0:
            residues_to_include = mod_rna_in_range.residues
            resindices = [res.resindex for res in residues_to_include]
            pocket_components['modified_rna'] = universe.select_atoms(
                f"resindex {' '.join(map(str, resindices))}"
            )
            print(f"Modified RNA: {len(mod_rna_in_range)} atoms → "
                  f"{len(pocket_components['modified_rna'])} atoms (complete residues)")

    # Protein pocket
    if len(all_components['protein']) > 0:
        protein_in_range = all_components['protein'].select_atoms(
            f"around {pocket_cutoff} global resname {ligand_resname}"
        )
        if len(protein_in_range) > 0:
            residues_to_include = protein_in_range.residues
            resindices = [res.resindex for res in residues_to_include]
            pocket_components['protein'] = universe.select_atoms(
                f"resindex {' '.join(map(str, resindices))}"
            )
            print(f"Protein: {len(protein_in_range)} atoms → "
                  f"{len(pocket_components['protein'])} atoms (complete residues)")

    # Target ligand (always include)
    pocket_components['ligand'] = ligand
    print(f"Ligand: {len(ligand)} atoms")

    return pocket_components


def safe_run_pdb4amber(input_pdb: Path, output_pdb: Path, options: str = "--dry --nohyd") -> Path:
    """
    Safely run pdb4amber with pre-checks to avoid crashes
    Returns output path (or input path if skipped/failed)
    """
    import parmed as pmd

    try:
        # Pre-check: count O5' and O3' atoms
        parm = pmd.load_file(str(input_pdb))
        o5_count = sum(1 for a in parm.atoms if a.name == "O5'" and a.residue.name in RNA_RESIDUES + MODIFIED_RNA)
        o3_count = sum(1 for a in parm.atoms if a.name == "O3'" and a.residue.name in RNA_RESIDUES + MODIFIED_RNA)

        if o5_count > o3_count + 1:
            print(f"  ⚠️  O5'({o5_count}) >> O3'({o3_count}), skipping pdb4amber (would crash)")
            return input_pdb

        # Run pdb4amber
        cmd = f"pdb4amber {options} -i {input_pdb} -o {output_pdb}"
        result = subprocess.run(
            cmd.split(),
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0 and output_pdb.exists():
            print(f"  ✓ pdb4amber cleaned {input_pdb.name}")
            return output_pdb
        else:
            print(f"  ⚠️  pdb4amber failed, using original file")
            return input_pdb

    except Exception as e:
        print(f"  ⚠️  pdb4amber error: {e}, using original file")
        return input_pdb


def clean_rna_terminal_atoms(input_pdb: Path, output_pdb: Path) -> bool:
    """
    Remove problematic terminal atoms from RNA fragments
    tleap will add them back with correct types

    Strategy:
    - Remove 5' phosphate groups (P, OP1, OP2, O5') from first residue
    - Remove 3' hydroxyl (O3') from last residue
    - Let tleap handle terminal capping
    """
    print(f"  Cleaning terminal atoms...")

    try:
        u = mda.Universe(str(input_pdb))

        if len(u.residues) == 0:
            print(f"    ⚠️  No residues found")
            return False

        # Get first and last RNA residues
        first_residue = u.residues[0]
        last_residue = u.residues[-1]

        # Atoms to remove
        atoms_to_remove = []

        # 5' terminal: remove phosphate group
        atoms_to_remove_5prime = {'P', 'OP1', 'OP2', 'O5P', 'O1P', 'O2P'}
        removed_5prime = []
        for atom in first_residue.atoms:
            if atom.name in atoms_to_remove_5prime:
                atoms_to_remove.append(atom)
                removed_5prime.append(atom.name)

        # 3' terminal: remove O3'
        atoms_to_remove_3prime = {"O3'", "O3*"}
        removed_3prime = []
        for atom in last_residue.atoms:
            if atom.name in atoms_to_remove_3prime:
                atoms_to_remove.append(atom)
                removed_3prime.append(atom.name)

        # Create new AtomGroup without these atoms
        if atoms_to_remove:
            remove_indices = {atom.index for atom in atoms_to_remove}
            keep_atoms = u.atoms[[i for i in range(len(u.atoms)) if i not in remove_indices]]

            # Save cleaned PDB
            keep_atoms.write(str(output_pdb))
            print(f"    ✓ Removed {len(atoms_to_remove)} terminal atoms ({len(removed_5prime)} from 5', {len(removed_3prime)} from 3')")
            return True
        else:
            print(f"    No terminal atoms to remove, copying original")
            import shutil
            shutil.copy(input_pdb, output_pdb)
            return True

    except Exception as e:
        print(f"    ✗ Error cleaning terminals: {e}")
        return False


def parameterize_rna(rna_atoms: mda.AtomGroup, output_prefix: Path) -> Tuple[bool, Optional[Path], Optional[Path]]:
    """
    Parameterize RNA using Amber RNA.OL3 force field
    Returns: (success, prmtop_path, inpcrd_path)
    """
    print(f"\n{'─'*70}")
    print(f"Parameterizing RNA: {len(rna_atoms)} atoms, {len(rna_atoms.residues)} residues")
    print(f"{'─'*70}")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Save RNA to PDB
    rna_pdb = output_prefix.parent / f"{output_prefix.stem}_rna.pdb"
    rna_atoms.write(str(rna_pdb))
    print(f"  Saved RNA to {rna_pdb.name}")

    # Clean terminal atoms to avoid tleap type errors
    # Remove 5' phosphate groups and 3' hydroxyl from pocket fragments
    # tleap will add them back with correct types
    cleaned_pdb = output_prefix.parent / f"{output_prefix.stem}_rna_cleaned.pdb"
    success = clean_rna_terminal_atoms(rna_pdb, cleaned_pdb)
    if not success:
        print(f"  ⚠️  Terminal cleaning failed, using original PDB")
        cleaned_pdb = rna_pdb

    # Generate tleap script
    tleap_script = output_prefix.parent / f"{output_prefix.stem}_rna_tleap.in"
    prmtop_file = output_prefix.parent / f"{output_prefix.stem}_rna.prmtop"
    inpcrd_file = output_prefix.parent / f"{output_prefix.stem}_rna.inpcrd"

    script_content = f"""source leaprc.RNA.OL3
mol = loadpdb {cleaned_pdb.name}

# For pocket fragments, don't try to cap terminals
# Just accept the structure as-is
set default nocenter on
set default PBRadii mbondi3

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
        cwd=str(output_prefix.parent),
        timeout=300
    )

    if prmtop_file.exists() and inpcrd_file.exists():
        print(f"  ✓ Successfully created {prmtop_file.name} and {inpcrd_file.name}")
        # Cleanup
        tleap_script.unlink()
        return True, prmtop_file, inpcrd_file
    else:
        print(f"  ✗ tleap failed")
        if result.stdout:
            print(f"  stdout: {result.stdout[-500:]}")
        return False, None, None


def parameterize_ligand_simple(ligand_atoms: mda.AtomGroup, ligand_name: str,
                               output_prefix: Path) -> Tuple[bool, Optional[Path], Optional[Path]]:
    """
    Simple ligand parameterization - save for future antechamber implementation
    For now, just save the ligand PDB for reference
    """
    print(f"\n{'─'*70}")
    print(f"Saving ligand: {len(ligand_atoms)} atoms")
    print(f"{'─'*70}")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Save ligand to PDB
    ligand_pdb = output_prefix.parent / f"{output_prefix.stem}_ligand_{ligand_name}.pdb"
    ligand_atoms.write(str(ligand_pdb))
    print(f"  Saved ligand to {ligand_pdb.name}")
    print(f"  ⚠️  Ligand parameterization not implemented yet")
    print(f"  TODO: Implement antechamber + GAFF workflow")

    return False, None, None


def parameterize_protein(protein_atoms: mda.AtomGroup, output_prefix: Path) -> Tuple[bool, Optional[Path], Optional[Path]]:
    """
    Parameterize protein using Amber ff14SB force field
    """
    print(f"\n{'─'*70}")
    print(f"Parameterizing protein: {len(protein_atoms)} atoms, {len(protein_atoms.residues)} residues")
    print(f"{'─'*70}")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Save protein to PDB
    protein_pdb = output_prefix.parent / f"{output_prefix.stem}_protein.pdb"
    protein_atoms.write(str(protein_pdb))
    print(f"  Saved protein to {protein_pdb.name}")

    # Try pdb4amber (usually more stable for proteins)
    cleaned_pdb = safe_run_pdb4amber(protein_pdb, protein_pdb.with_suffix('.cleaned.pdb'))

    # Generate tleap script
    tleap_script = output_prefix.parent / f"{output_prefix.stem}_protein_tleap.in"
    prmtop_file = output_prefix.parent / f"{output_prefix.stem}_protein.prmtop"
    inpcrd_file = output_prefix.parent / f"{output_prefix.stem}_protein.inpcrd"

    script_content = f"""source leaprc.protein.ff14SB
mol = loadpdb {cleaned_pdb.name}
set default nocenter on
set default PBRadii mbondi3
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
        cwd=str(output_prefix.parent),
        timeout=300
    )

    if prmtop_file.exists() and inpcrd_file.exists():
        print(f"  ✓ Successfully created {prmtop_file.name} and {inpcrd_file.name}")
        tleap_script.unlink()
        return True, prmtop_file, inpcrd_file
    else:
        print(f"  ✗ tleap failed")
        return False, None, None


def process_complex_v2(pdb_id: str, ligand_name: str, hariboss_dir: Path,
                      output_dir: Path, pocket_cutoff: float) -> Dict:
    """
    Process a single complex with new strategy
    Returns dict with processing results
    """
    print(f"\n{'='*70}")
    print(f"Processing: {pdb_id} - {ligand_name}")
    print(f"{'='*70}")

    result = {
        'pdb_id': pdb_id,
        'ligand': ligand_name,
        'success': False,
        'components': {},
        'errors': []
    }

    # Load structure - look in data/raw/mmCIF first, then hariboss_dir
    cif_file = Path("data/raw/mmCIF") / f"{pdb_id}.cif"
    if not cif_file.exists():
        cif_file = hariboss_dir / "mmCIF" / f"{pdb_id}.cif"

    if not cif_file.exists():
        result['errors'].append(f"CIF file not found in data/raw/mmCIF or {hariboss_dir}/mmCIF")
        return result

    try:
        # Convert CIF to PDB (temp file)
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(cif_file))

        temp_pdb = output_dir / "temp" / f"{pdb_id}_temp.pdb"
        temp_pdb.parent.mkdir(parents=True, exist_ok=True)

        io = PDBIO()
        io.set_structure(structure)
        io.save(str(temp_pdb))

        # Load with MDAnalysis
        u = mda.Universe(str(temp_pdb))

        # Define pocket with separated components
        pocket_components = define_pocket_by_residues(u, ligand_name, pocket_cutoff)

        if not pocket_components:
            result['errors'].append("No pocket components found")
            return result

        # Save combined pocket for reference
        all_pocket_atoms = None
        for comp_atoms in pocket_components.values():
            if all_pocket_atoms is None:
                all_pocket_atoms = comp_atoms
            else:
                all_pocket_atoms = all_pocket_atoms + comp_atoms

        if all_pocket_atoms and len(all_pocket_atoms) > 0:
            pocket_pdb = output_dir / "pockets" / f"{pdb_id}_{ligand_name}_pocket.pdb"
            pocket_pdb.parent.mkdir(parents=True, exist_ok=True)
            all_pocket_atoms.write(str(pocket_pdb))
            print(f"\n✓ Saved combined pocket: {len(all_pocket_atoms)} atoms")

        # Parameterize each component separately
        output_prefix = output_dir / "amber" / f"{pdb_id}_{ligand_name}"
        output_prefix.parent.mkdir(parents=True, exist_ok=True)

        # RNA
        if 'rna' in pocket_components and len(pocket_components['rna']) > 0:
            success, prmtop, inpcrd = parameterize_rna(
                pocket_components['rna'],
                output_prefix
            )
            result['components']['rna'] = {
                'success': success,
                'atoms': len(pocket_components['rna']),
                'residues': len(pocket_components['rna'].residues),
                'prmtop': str(prmtop) if prmtop else None,
                'inpcrd': str(inpcrd) if inpcrd else None
            }

        # Ligand
        if 'ligand' in pocket_components and len(pocket_components['ligand']) > 0:
            success, prmtop, inpcrd = parameterize_ligand_simple(
                pocket_components['ligand'],
                ligand_name,
                output_prefix
            )
            result['components']['ligand'] = {
                'success': success,
                'atoms': len(pocket_components['ligand']),
                'saved': True
            }

        # Protein
        if 'protein' in pocket_components and len(pocket_components['protein']) > 0:
            success, prmtop, inpcrd = parameterize_protein(
                pocket_components['protein'],
                output_prefix
            )
            result['components']['protein'] = {
                'success': success,
                'atoms': len(pocket_components['protein']),
                'residues': len(pocket_components['protein'].residues),
                'prmtop': str(prmtop) if prmtop else None,
                'inpcrd': str(inpcrd) if inpcrd else None
            }

        # Check if at least RNA was successful
        if 'rna' in result['components'] and result['components']['rna']['success']:
            result['success'] = True

        # Cleanup temp file
        temp_pdb.unlink()

    except Exception as e:
        result['errors'].append(f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Process RNA-ligand complexes with separated parameterization (V2)"
    )
    parser.add_argument("--hariboss_csv", type=str, required=True,
                       help="Path to HARIBOSS complexes CSV file")
    parser.add_argument("--hariboss_dir", type=str, default="hariboss",
                       help="Directory containing HARIBOSS data (with mmCIF subdirectory)")
    parser.add_argument("--output_dir", type=str, default="data",
                       help="Base output directory")
    parser.add_argument("--pocket_cutoff", type=float, default=5.0,
                       help="Cutoff distance (Å) for pocket definition")
    parser.add_argument("--max_complexes", type=int, default=None,
                       help="Maximum number of complexes to process (for testing)")

    args = parser.parse_args()

    # Read HARIBOSS CSV
    print(f"Reading HARIBOSS CSV from {args.hariboss_csv}...")
    df = pd.read_csv(args.hariboss_csv)
    print(f"Found {len(df)} complexes in HARIBOSS dataset")

    # Limit for testing
    if args.max_complexes:
        df = df.head(args.max_complexes)
        print(f"Processing first {args.max_complexes} complexes")

    # Setup paths
    hariboss_dir = Path(args.hariboss_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each complex
    results = []
    failed = []

    print(f"\n{'='*70}")
    print(f"Processing {len(df)} complexes")
    print(f"{'='*70}\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing complexes"):
        pdb_id = row['id'].lower()

        # Parse ligands
        import ast
        ligands_str = row['sm_ligand_ids']
        try:
            ligands = ast.literal_eval(ligands_str)
            if not isinstance(ligands, list):
                ligands = [ligands_str]
        except:
            ligands = [ligands_str]

        # Process first ligand
        if ligands and len(ligands) > 0:
            # Format: "ARG_.:B/1:N" -> extract "ARG"
            ligand_info = ligands[0].split('_')[0].split(':')[0]

            result = process_complex_v2(
                pdb_id, ligand_info,
                hariboss_dir, output_dir,
                args.pocket_cutoff
            )

            results.append(result)

            if not result['success']:
                failed.append({
                    'pdb_id': pdb_id,
                    'ligand': ligand_info,
                    'errors': result['errors']
                })

    # Save results
    results_file = output_dir / "processing_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total complexes: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\nFailed complexes:")
        for f in failed[:10]:  # Show first 10
            print(f"  {f['pdb_id']} ({f['ligand']}): {', '.join(f['errors'])}")

        # Save failed list
        failed_df = pd.DataFrame(failed)
        failed_file = output_dir / "failed_complexes_v2.csv"
        failed_df.to_csv(failed_file, index=False)
        print(f"\nFailed complexes saved to {failed_file}")

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
