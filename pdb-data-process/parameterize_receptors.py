#!/usr/bin/env python3
"""
Parameterize receptors with AMBER force fields based on their classification.
Supports RNA, DNA, Protein, and Complex types with appropriate force fields.
"""
import os
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
import argparse
import MDAnalysis as mda


RNA_RESIDUES = {'A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT'}
PROTEIN_RESIDUES = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
                    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
                    'TYR', 'VAL', 'HID', 'HIE', 'HIP', 'CYX'}


def classify_receptor(pdb_file):
    """Classify receptor as RNA, Protein, or Complex."""
    u = mda.Universe(str(pdb_file))
    residues = set(res.resname for res in u.residues)

    has_rna = bool(residues & RNA_RESIDUES)
    has_protein = bool(residues & PROTEIN_RESIDUES)

    if has_rna and has_protein:
        return 'Complex', residues
    elif has_rna:
        return 'RNA', residues
    elif has_protein:
        return 'Protein', residues
    else:
        return 'Unknown', residues


def parameterize_rna(pdb_file, output_prefix):
    """Parameterize RNA/DNA with RNA.OL3 force field."""
    tleap_script = output_prefix.parent / f"{output_prefix.stem}_tleap.in"
    prmtop_file = output_prefix.with_suffix('.prmtop')
    inpcrd_file = output_prefix.with_suffix('.inpcrd')

    script_content = f"""source leaprc.RNA.OL3
mol = loadpdb {pdb_file.name}
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

    tleap_script.unlink()

    if prmtop_file.exists() and inpcrd_file.exists():
        return True, prmtop_file, inpcrd_file
    else:
        return False, None, None


def parameterize_protein(pdb_file, output_prefix):
    """Parameterize protein with ff14SB force field."""
    tleap_script = output_prefix.parent / f"{output_prefix.stem}_tleap.in"
    prmtop_file = output_prefix.with_suffix('.prmtop')
    inpcrd_file = output_prefix.with_suffix('.inpcrd')

    script_content = f"""source leaprc.protein.ff14SB
mol = loadpdb {pdb_file.name}
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

    tleap_script.unlink()

    if prmtop_file.exists() and inpcrd_file.exists():
        return True, prmtop_file, inpcrd_file
    else:
        return False, None, None


def parameterize_complex(pdb_file, output_prefix):
    """Parameterize complex with both RNA.OL3 and ff14SB."""
    tleap_script = output_prefix.parent / f"{output_prefix.stem}_tleap.in"
    prmtop_file = output_prefix.with_suffix('.prmtop')
    inpcrd_file = output_prefix.with_suffix('.inpcrd')

    script_content = f"""source leaprc.RNA.OL3
source leaprc.protein.ff14SB
mol = loadpdb {pdb_file.name}
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

    tleap_script.unlink()

    if prmtop_file.exists() and inpcrd_file.exists():
        return True, prmtop_file, inpcrd_file
    else:
        return False, None, None


def process_single_file(args):
    """Process a single receptor file."""
    pdb_file, receptor_type, output_dir = args

    try:
        pdb_id = pdb_file.stem.replace('_polymer_fixed', '')
        output_prefix = output_dir / "amber" / pdb_id
        output_prefix.parent.mkdir(parents=True, exist_ok=True)

        if receptor_type == 'RNA':
            success, prmtop, inpcrd = parameterize_rna(pdb_file, output_prefix)
        elif receptor_type == 'Protein':
            success, prmtop, inpcrd = parameterize_protein(pdb_file, output_prefix)
        elif receptor_type == 'Complex':
            success, prmtop, inpcrd = parameterize_complex(pdb_file, output_prefix)
        else:
            return {
                'pdb_id': pdb_id,
                'type': receptor_type,
                'status': 'unknown_type',
                'prmtop': None,
                'inpcrd': None
            }

        return {
            'pdb_id': pdb_id,
            'type': receptor_type,
            'status': 'success' if success else 'failed',
            'prmtop': str(prmtop) if prmtop else None,
            'inpcrd': str(inpcrd) if inpcrd else None
        }

    except Exception as e:
        return {
            'pdb_id': pdb_file.stem,
            'type': receptor_type,
            'status': f'error: {str(e)}',
            'prmtop': None,
            'inpcrd': None
        }


def main():
    parser = argparse.ArgumentParser(description='Parameterize receptors with AMBER')
    parser.add_argument('--receptor-dir', default='effect_receptor',
                        help='Directory containing receptor files')
    parser.add_argument('--output-dir', default='effect_receptor_processed',
                        help='Output directory')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes')

    args = parser.parse_args()

    receptor_dir = Path(args.receptor_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all PDB files with their types
    process_args = []

    for receptor_type in ['RNA', 'Protein', 'Complex']:
        type_dir = receptor_dir / receptor_type
        if not type_dir.exists():
            continue

        for pdb_file in type_dir.glob('*.pdb'):
            process_args.append((pdb_file, receptor_type, output_dir))

    if not process_args:
        print("No PDB files found!")
        return

    print(f"Found {len(process_args)} receptor files to process")

    num_workers = args.workers or cpu_count()
    print(f"Processing with {num_workers} workers...")

    with Pool(num_workers) as pool:
        results = pool.map(process_single_file, process_args)

    # Statistics
    success_count = sum(1 for r in results if r['status'] == 'success')

    print(f"\n{'='*70}")
    print(f"PARAMETERIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total: {len(results)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(results) - success_count}")

    # Group by type
    by_type = {}
    for r in results:
        t = r['type']
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(r)

    for t, items in by_type.items():
        success = sum(1 for i in items if i['status'] == 'success')
        print(f"\n{t}: {success}/{len(items)} successful")


if __name__ == '__main__':
    main()
