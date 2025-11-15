#!/usr/bin/env python3
"""
Extract receptor pockets around ligands and parameterize with AMBER.
Includes terminal cleaning for RNA, Protein, and Complex.
Based on scripts/01_process_data.py
"""
import os
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
import argparse
import MDAnalysis as mda
import parmed as pmd
from typing import Dict, Optional, Tuple
import json
RNA_RESIDUES = ['A', 'C', 'G', 'U', 'A3', 'A5', 'C3', 'C5', 'G3', 'G5', 'U3', 'U5']
DNA_RESIDUES = ['DA', 'DC', 'DG', 'DT', 'DA3', 'DA5', 'DC3', 'DC5', 'DG3', 'DG5', 'DT3', 'DT5']
MODIFIED_RNA = ['PSU', '5MU', '5MC', '1MA', '7MG', 'M2G', 'OMC', 'OMG', 'H2U',
                '2MG', 'M7G', 'OMU', 'YYG', 'YG', '6MZ', 'IU', 'I']
PROTEIN_RESIDUES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
                    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
                    'TYR', 'VAL', 'HID', 'HIE', 'HIP', 'CYX']
def extract_ligand_info(ligand_pdb):
    """Extract ligand residue name from PDB file."""
    with open(ligand_pdb, 'r') as f:
        for line in f:
            if line.startswith('HETATM') or line.startswith('ATOM'):
                return line[17:20].strip()
    return None
def define_pocket_by_residues(receptor_u, ligand_resname, pocket_cutoff=5.0):
    """Define pocket using complete residues within cutoff of ligand."""
    ligand = receptor_u.select_atoms(f"resname {ligand_resname}")
    if len(ligand) == 0:
        return None
    all_atoms = receptor_u.atoms
    receptor_atoms = all_atoms - ligand
    nearby_atoms = receptor_atoms.select_atoms(
        f"around {pocket_cutoff} global resname {ligand_resname}"
    )
    if len(nearby_atoms) == 0:
        return None
    residues_to_include = nearby_atoms.residues
    resindices = [res.resindex for res in residues_to_include]
    pocket_atoms = receptor_u.select_atoms(f"resindex {' '.join(map(str, resindices))}")
    pocket_components = {}
    nucleic_residues = [res for res in pocket_atoms.residues if res.resname in RNA_RESIDUES + DNA_RESIDUES]
    if nucleic_residues:
        nucleic_resindices = [res.resindex for res in nucleic_residues]
        pocket_components['nucleic'] = receptor_u.select_atoms(f"resindex {' '.join(map(str, nucleic_resindices))}")
    mod_rna_residues = [res for res in pocket_atoms.residues if res.resname in MODIFIED_RNA]
    if mod_rna_residues:
        mod_rna_resindices = [res.resindex for res in mod_rna_residues]
        pocket_components['modified_rna'] = receptor_u.select_atoms(f"resindex {' '.join(map(str, mod_rna_resindices))}")
    protein_residues = [res for res in pocket_atoms.residues if res.resname in PROTEIN_RESIDUES]
    if protein_residues:
        protein_resindices = [res.resindex for res in protein_residues]
        pocket_components['protein'] = receptor_u.select_atoms(f"resindex {' '.join(map(str, protein_resindices))}")
    return pocket_components
def safe_run_pdb4amber(input_pdb, output_pdb, options="--dry --nohyd"):
    """Safely run pdb4amber with pre-checks."""
    try:
        parm = pmd.load_file(str(input_pdb))
        o5_count = sum(1 for a in parm.atoms if a.name == "O5'" and a.residue.name in RNA_RESIDUES + DNA_RESIDUES + MODIFIED_RNA)
        o3_count = sum(1 for a in parm.atoms if a.name == "O3'" and a.residue.name in RNA_RESIDUES + DNA_RESIDUES + MODIFIED_RNA)
        if o5_count > o3_count + 1:
            return input_pdb
        cmd = f"pdb4amber {options} -i {input_pdb} -o {output_pdb}"
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and output_pdb.exists():
            return output_pdb
        else:
            return input_pdb
    except:
        return input_pdb
def clean_nucleic_terminal_atoms(input_pdb, output_pdb):
    """Remove problematic terminal atoms from RNA/DNA fragments."""
    try:
        u = mda.Universe(str(input_pdb))
        if len(u.residues) == 0:
            return False
        chains = {}
        for residue in u.residues:
            if residue.resname not in RNA_RESIDUES + DNA_RESIDUES + MODIFIED_RNA:
                continue
            chain_id = residue.segid if residue.segid else 'X'
            if chain_id not in chains:
                chains[chain_id] = []
            chains[chain_id].append(residue)
        atoms_to_remove = []
        atoms_to_remove_5prime = {'P', 'OP1', 'OP2', 'O5P', 'O1P', 'O2P'}
        atoms_to_remove_3prime = {"O3'", "O3*"}
        for chain_id, residues in chains.items():
            if len(residues) == 0:
                continue
            first_residue = residues[0]
            for atom in first_residue.atoms:
                if atom.name in atoms_to_remove_5prime:
                    atoms_to_remove.append(atom)
            last_residue = residues[-1]
            for atom in last_residue.atoms:
                if atom.name in atoms_to_remove_3prime:
                    atoms_to_remove.append(atom)
        if atoms_to_remove:
            remove_indices = {atom.index for atom in atoms_to_remove}
            keep_atoms = u.atoms[[i for i in range(len(u.atoms)) if i not in remove_indices]]
            keep_atoms.write(str(output_pdb))
            return True
        else:
            import shutil
            shutil.copy(input_pdb, output_pdb)
            return True
    except:
        return False
def clean_protein_terminal_atoms(input_pdb, output_pdb):
    """Remove problematic terminal atoms from protein fragments."""
    try:
        u = mda.Universe(str(input_pdb))
        if len(u.residues) == 0:
            return False
        chains = {}
        for residue in u.residues:
            if residue.resname not in PROTEIN_RESIDUES:
                continue
            chain_id = residue.segid if residue.segid else 'X'
            if chain_id not in chains:
                chains[chain_id] = []
            chains[chain_id].append(residue)
        atoms_to_remove = []
        atoms_to_remove_nterm = {'OXT', 'HXT'}
        atoms_to_remove_cterm = {'OXT', 'HXT'}
        for chain_id, residues in chains.items():
            if len(residues) == 0:
                continue
            first_residue = residues[0]
            for atom in first_residue.atoms:
                if atom.name in atoms_to_remove_nterm:
                    atoms_to_remove.append(atom)
            last_residue = residues[-1]
            for atom in last_residue.atoms:
                if atom.name in atoms_to_remove_cterm:
                    atoms_to_remove.append(atom)
        if atoms_to_remove:
            remove_indices = {atom.index for atom in atoms_to_remove}
            keep_atoms = u.atoms[[i for i in range(len(u.atoms)) if i not in remove_indices]]
            keep_atoms.write(str(output_pdb))
            return True
        else:
            import shutil
            shutil.copy(input_pdb, output_pdb)
            return True
    except:
        return False
def clean_complex_terminal_atoms(input_pdb, output_pdb):
    """Remove problematic terminal atoms from both RNA/DNA and protein in complex."""
    try:
        u = mda.Universe(str(input_pdb))
        if len(u.residues) == 0:
            return False
        nucleic_chains = {}
        protein_chains = {}
        for residue in u.residues:
            chain_id = residue.segid if residue.segid else 'X'
            if residue.resname in RNA_RESIDUES + DNA_RESIDUES + MODIFIED_RNA:
                if chain_id not in nucleic_chains:
                    nucleic_chains[chain_id] = []
                nucleic_chains[chain_id].append(residue)
            elif residue.resname in PROTEIN_RESIDUES:
                if chain_id not in protein_chains:
                    protein_chains[chain_id] = []
                protein_chains[chain_id].append(residue)
        atoms_to_remove = []
        # Clean nucleic acid terminals
        atoms_to_remove_5prime = {'P', 'OP1', 'OP2', 'O5P', 'O1P', 'O2P'}
        atoms_to_remove_3prime = {"O3'", "O3*"}
        for chain_id, residues in nucleic_chains.items():
            if len(residues) == 0:
                continue
            first_residue = residues[0]
            for atom in first_residue.atoms:
                if atom.name in atoms_to_remove_5prime:
                    atoms_to_remove.append(atom)
            last_residue = residues[-1]
            for atom in last_residue.atoms:
                if atom.name in atoms_to_remove_3prime:
                    atoms_to_remove.append(atom)
        # Clean protein terminals
        atoms_to_remove_nterm = {'OXT', 'HXT'}
        atoms_to_remove_cterm = {'OXT', 'HXT'}
        for chain_id, residues in protein_chains.items():
            if len(residues) == 0:
                continue
            first_residue = residues[0]
            for atom in first_residue.atoms:
                if atom.name in atoms_to_remove_nterm:
                    atoms_to_remove.append(atom)
            last_residue = residues[-1]
            for atom in last_residue.atoms:
                if atom.name in atoms_to_remove_cterm:
                    atoms_to_remove.append(atom)
        if atoms_to_remove:
            remove_indices = {atom.index for atom in atoms_to_remove}
            keep_atoms = u.atoms[[i for i in range(len(u.atoms)) if i not in remove_indices]]
            keep_atoms.write(str(output_pdb))
            return True
        else:
            import shutil
            shutil.copy(input_pdb, output_pdb)
            return True
    except:
        return False
def classify_nucleic_acid_type(residues):
    """Classify nucleic acid as DNA, RNA, or HYBRID based on residues."""
    has_rna = any(res.resname in RNA_RESIDUES for res in residues)
    has_dna = any(res.resname in DNA_RESIDUES for res in residues)

    if has_dna and not has_rna:
        return 'DNA'
    elif has_rna and not has_dna:
        return 'RNA'
    elif has_dna and has_rna:
        return 'HYBRID'
    else:
        return None
def parameterize_nucleic(nucleic_atoms, output_prefix):
    """Parameterize nucleic acid pocket with appropriate force field (DNA.OL15 or RNA.OL3)."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Classify as DNA, RNA, or hybrid
    nucleic_type = classify_nucleic_acid_type(nucleic_atoms.residues)

    nucleic_pdb = output_prefix.parent / f"{output_prefix.stem}_nucleic.pdb"
    nucleic_atoms.write(str(nucleic_pdb))

    cleaned_pdb = output_prefix.parent / f"{output_prefix.stem}_nucleic_cleaned.pdb"
    success = clean_nucleic_terminal_atoms(nucleic_pdb, cleaned_pdb)
    if not success:
        cleaned_pdb = nucleic_pdb

    tleap_script = output_prefix.parent / f"{output_prefix.stem}_nucleic_tleap.in"
    prmtop_file = output_prefix.parent / f"{output_prefix.stem}_nucleic.prmtop"
    inpcrd_file = output_prefix.parent / f"{output_prefix.stem}_nucleic.inpcrd"
    leap_log = output_prefix.parent / f"{output_prefix.stem}_nucleic_leap.log"

    # Select force field based on nucleic acid type
    if nucleic_type == 'DNA':
        leaprc = "source leaprc.DNA.OL15"
    elif nucleic_type == 'RNA':
        leaprc = "source leaprc.RNA.OL3"
    else:  # HYBRID
        leaprc = "source leaprc.DNA.OL15\nsource leaprc.RNA.OL3"

    script_content = f"""{leaprc}
mol = loadpdb {cleaned_pdb.name}
set default nocenter on
set default PBRadii mbondi3
saveamberparm mol {prmtop_file.name} {inpcrd_file.name}
quit
"""
    tleap_script.write_text(script_content)

    result = subprocess.run(
        ["tleap", "-f", tleap_script.name],
        capture_output=True,
        text=True,
        cwd=str(output_prefix.parent),
        timeout=300
    )

    # Save leap log for debugging
    leap_log.write_text(result.stdout + "\n" + result.stderr)

    tleap_script.unlink()

    if prmtop_file.exists() and inpcrd_file.exists():
        return True, prmtop_file, inpcrd_file
    else:
        return False, None, None
def parameterize_protein(protein_atoms, output_prefix):
    """Parameterize protein pocket with ff14SB."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    protein_pdb = output_prefix.parent / f"{output_prefix.stem}_protein.pdb"
    protein_atoms.write(str(protein_pdb))

    cleaned_pdb = output_prefix.parent / f"{output_prefix.stem}_protein_cleaned.pdb"
    success = clean_protein_terminal_atoms(protein_pdb, cleaned_pdb)
    if not success:
        cleaned_pdb = safe_run_pdb4amber(protein_pdb, protein_pdb.with_suffix('.cleaned.pdb'))

    tleap_script = output_prefix.parent / f"{output_prefix.stem}_protein_tleap.in"
    prmtop_file = output_prefix.parent / f"{output_prefix.stem}_protein.prmtop"
    inpcrd_file = output_prefix.parent / f"{output_prefix.stem}_protein.inpcrd"
    leap_log = output_prefix.parent / f"{output_prefix.stem}_protein_leap.log"

    script_content = f"""source leaprc.protein.ff14SB
set default nocap on
mol = loadpdb {cleaned_pdb.name}
set default nocenter on
set default PBRadii mbondi3
saveamberparm mol {prmtop_file.name} {inpcrd_file.name}
quit
"""
    tleap_script.write_text(script_content)

    result = subprocess.run(
        ["tleap", "-f", tleap_script.name],
        capture_output=True,
        text=True,
        cwd=str(output_prefix.parent),
        timeout=300
    )

    # Save leap log for debugging
    leap_log.write_text(result.stdout + "\n" + result.stderr)

    tleap_script.unlink()

    if prmtop_file.exists() and inpcrd_file.exists():
        return True, prmtop_file, inpcrd_file
    else:
        return False, None, None
def parameterize_complex(nucleic_atoms, protein_atoms, output_prefix):
    """Parameterize complex pocket with appropriate nucleic acid and protein force fields."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Classify nucleic acid type
    nucleic_type = classify_nucleic_acid_type(nucleic_atoms.residues)

    # Combine nucleic and protein
    combined_atoms = nucleic_atoms + protein_atoms
    complex_pdb = output_prefix.parent / f"{output_prefix.stem}_complex.pdb"
    combined_atoms.write(str(complex_pdb))

    cleaned_pdb = output_prefix.parent / f"{output_prefix.stem}_complex_cleaned.pdb"
    success = clean_complex_terminal_atoms(complex_pdb, cleaned_pdb)
    if not success:
        cleaned_pdb = complex_pdb

    tleap_script = output_prefix.parent / f"{output_prefix.stem}_complex_tleap.in"
    prmtop_file = output_prefix.parent / f"{output_prefix.stem}_complex.prmtop"
    inpcrd_file = output_prefix.parent / f"{output_prefix.stem}_complex.inpcrd"
    leap_log = output_prefix.parent / f"{output_prefix.stem}_complex_leap.log"

    # Select force field based on nucleic acid type
    if nucleic_type == 'DNA':
        leaprc_nucleic = "source leaprc.DNA.OL15"
    elif nucleic_type == 'RNA':
        leaprc_nucleic = "source leaprc.RNA.OL3"
    else:  # HYBRID
        leaprc_nucleic = "source leaprc.DNA.OL15\nsource leaprc.RNA.OL3"

    script_content = f"""{leaprc_nucleic}
source leaprc.protein.ff14SB
set default nocap on
mol = loadpdb {cleaned_pdb.name}
set default nocenter on
set default PBRadii mbondi3
saveamberparm mol {prmtop_file.name} {inpcrd_file.name}
quit
"""
    tleap_script.write_text(script_content)

    result = subprocess.run(
        ["tleap", "-f", tleap_script.name],
        capture_output=True,
        text=True,
        cwd=str(output_prefix.parent),
        timeout=300
    )

    # Save leap log for debugging
    leap_log.write_text(result.stdout + "\n" + result.stderr)

    tleap_script.unlink()

    if prmtop_file.exists() and inpcrd_file.exists():
        return True, prmtop_file, inpcrd_file
    else:
        return False, None, None
def merge_receptor_and_ligand(receptor_pdb, ligand_pdb, output_pdb):
    """Merge receptor and ligand into single PDB for pocket definition."""
    receptor_u = mda.Universe(str(receptor_pdb))
    ligand_u = mda.Universe(str(ligand_pdb))
    merged = mda.Merge(receptor_u.atoms, ligand_u.atoms)
    merged.atoms.write(str(output_pdb))
    return merged
def process_single_complex(args):
    """Process a single ligand-receptor pair to extract and parameterize pocket."""
    ligand_pdb, receptor_dir, output_dir, pocket_cutoff = args
    try:
        ligand_filename = ligand_pdb.stem
        pdb_id = ligand_filename.replace('_ligands', '')
        ligand_name = extract_ligand_info(ligand_pdb)
        if not ligand_name:
            return {'pdb_id': pdb_id, 'status': 'no_ligand_found'}
        receptor_file = None
        for receptor_type in ['RNA', 'Protein', 'Complex']:
            candidate = receptor_dir / receptor_type / f"{pdb_id}_polymer_fixed.pdb"
            if candidate.exists():
                receptor_file = candidate
                break
        if not receptor_file:
            return {'pdb_id': pdb_id, 'status': 'receptor_not_found'}
        temp_dir = output_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        merged_pdb = temp_dir / f"{pdb_id}_merged.pdb"
        merged_u = merge_receptor_and_ligand(receptor_file, ligand_pdb, merged_pdb)
        pocket_components = define_pocket_by_residues(merged_u, ligand_name, pocket_cutoff)
        if not pocket_components:
            return {'pdb_id': pdb_id, 'status': 'pocket_empty'}
        pocket_dir = output_dir / "pockets"
        pocket_dir.mkdir(parents=True, exist_ok=True)
        all_pocket_atoms = None
        for comp_atoms in pocket_components.values():
            if all_pocket_atoms is None:
                all_pocket_atoms = comp_atoms
            else:
                all_pocket_atoms = all_pocket_atoms + comp_atoms
        if all_pocket_atoms:
            pocket_pdb = pocket_dir / f"{pdb_id}_pocket.pdb"
            all_pocket_atoms.write(str(pocket_pdb))
        output_prefix = output_dir / "amber" / pdb_id
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        result = {
            'pdb_id': pdb_id,
            'ligand_name': ligand_name,
            'status': 'success',
            'components': {}
        }
        has_nucleic = 'nucleic' in pocket_components or 'modified_rna' in pocket_components
        has_protein = 'protein' in pocket_components
        # Parameterize based on pocket composition
        if has_nucleic and has_protein:
            # Complex: combine nucleic acid and protein
            nucleic_atoms = pocket_components.get('nucleic', mda.AtomGroup([], merged_u))
            if 'modified_rna' in pocket_components:
                nucleic_atoms = nucleic_atoms + pocket_components['modified_rna']
            protein_atoms = pocket_components['protein']
            success, prmtop, inpcrd = parameterize_complex(nucleic_atoms, protein_atoms, output_prefix)
            result['components']['complex'] = {
                'success': success,
                'nucleic_atoms': len(nucleic_atoms),
                'protein_atoms': len(protein_atoms),
                'prmtop': str(prmtop) if prmtop else None,
                'inpcrd': str(inpcrd) if inpcrd else None
            }
        elif has_nucleic:
            # Nucleic acid only (RNA or DNA)
            nucleic_atoms = pocket_components.get('nucleic', mda.AtomGroup([], merged_u))
            if 'modified_rna' in pocket_components:
                nucleic_atoms = nucleic_atoms + pocket_components['modified_rna']
            success, prmtop, inpcrd = parameterize_nucleic(nucleic_atoms, output_prefix)
            result['components']['nucleic'] = {
                'success': success,
                'atoms': len(nucleic_atoms),
                'prmtop': str(prmtop) if prmtop else None,
                'inpcrd': str(inpcrd) if inpcrd else None
            }
        elif has_protein:
            # Protein only
            success, prmtop, inpcrd = parameterize_protein(pocket_components['protein'], output_prefix)
            result['components']['protein'] = {
                'success': success,
                'atoms': len(pocket_components['protein']),
                'prmtop': str(prmtop) if prmtop else None,
                'inpcrd': str(inpcrd) if inpcrd else None
            }
        if merged_pdb.exists():
            merged_pdb.unlink()
        return result
    except Exception as e:
        import traceback
        return {
            'pdb_id': ligand_pdb.stem,
            'status': f'error: {str(e)}',
            'traceback': traceback.format_exc()
        }
def main():
    parser = argparse.ArgumentParser(description='Extract pockets and parameterize with AMBER')
    parser.add_argument('--ligand-dir', default='extracted_ligands/processed_ligands_effect_1')
    parser.add_argument('--receptor-dir', default='effect_receptor')
    parser.add_argument('--output-dir', default='processed_output')
    parser.add_argument('--pocket-cutoff', type=float, default=5.0)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    ligand_dir = Path(args.ligand_dir)
    receptor_dir = Path(args.receptor_dir)
    output_dir = Path(args.output_dir)
    ligand_files = sorted(ligand_dir.glob('*.pdb'))
    if not ligand_files:
        print(f"No ligand files found in {ligand_dir}")
        return
    print(f"Found {len(ligand_files)} ligand files")
    print(f"Pocket cutoff: {args.pocket_cutoff} Å")
    process_args = [(f, receptor_dir, output_dir, args.pocket_cutoff) for f in ligand_files]
    num_workers = args.workers or cpu_count()
    print(f"Processing with {num_workers} workers...\n")
    with Pool(num_workers) as pool:
        results = pool.map(process_single_complex, process_args)
    success_count = sum(1 for r in results if r.get('status') == 'success')
    print(f"\n{'='*70}")
    print(f"POCKET EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total: {len(results)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(results) - success_count}\n")
    nucleic_count = sum(1 for r in results if 'nucleic' in r.get('components', {}))
    protein_count = sum(1 for r in results if 'protein' in r.get('components', {}))
    complex_count = sum(1 for r in results if 'complex' in r.get('components', {}))
    print(f"Nucleic acid pockets: {nucleic_count}")
    print(f"Protein pockets: {protein_count}")
    print(f"Complex pockets: {complex_count}")
    results_file = output_dir / 'pocket_parameterization_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
if __name__ == '__main__':
    main()