#!/usr/bin/env python3
"""
Generate PyTorch Geometric graph from RNA binding pocket

This script takes an RNA binding pocket PDB file and generates a molecular graph
that can be used as input to the RNA-3E-FFI model.

Usage:
    python generate_pocket_graph.py --input pocket.pdb --output pocket_graph.pt
    python generate_pocket_graph.py --input pocket.pdb --output pocket_graph.pt --ligand_resname ATP
"""

import argparse
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
from torch_geometric.data import Data
import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem import AllChem
import parmed as pmd
import tempfile
import shutil


# RNA residue definitions (same as 01_process_data.py)
RNA_RESIDUES = ['A', 'C', 'G', 'U', 'A3', 'A5', 'C3', 'C5', 'G3', 'G5', 'U3', 'U5',
                'DA', 'DC', 'DG', 'DT', 'DA3', 'DA5', 'DC3', 'DC5', 'DG3', 'DG5', 'DT3', 'DT5']

MODIFIED_RNA = ['PSU', '5MU', '5MC', '1MA', '7MG', 'M2G', 'OMC', 'OMG', 'H2U',
                '2MG', 'M7G', 'OMU', 'YYG', 'YG', '6MZ', 'IU', 'I']


def clean_rna_terminal_atoms(input_pdb: Path, output_pdb: Path) -> bool:
    """
    Remove problematic terminal atoms from RNA fragments.
    tleap will add them back with correct types.
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
            print(f"    ✓ Removed {len(atoms_to_remove)} terminal atoms")
            return True
        else:
            print(f"    No terminal atoms to remove, copying original")
            shutil.copy(input_pdb, output_pdb)
            return True

    except Exception as e:
        print(f"    ✗ Error cleaning terminals: {e}")
        return False


def parameterize_rna_pocket(rna_pdb: Path, output_dir: Path) -> Tuple[bool, Optional[Path], Optional[Path]]:
    """
    Parameterize RNA pocket using Amber RNA.OL3 force field.

    Args:
        rna_pdb: Path to RNA PDB file
        output_dir: Directory for output files

    Returns:
        (success, prmtop_path, inpcrd_path)
    """
    print(f"\n{'─'*70}")
    print(f"Parameterizing RNA pocket with Amber force field")
    print(f"{'─'*70}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and check RNA
    try:
        u = mda.Universe(str(rna_pdb))
        n_atoms = len(u.atoms)
        n_residues = len(u.residues)
        print(f"  RNA: {n_atoms} atoms, {n_residues} residues")
    except Exception as e:
        print(f"  ✗ Failed to load RNA PDB: {e}")
        return False, None, None

    # Clean terminal atoms
    cleaned_pdb = output_dir / "rna_cleaned.pdb"
    success = clean_rna_terminal_atoms(rna_pdb, cleaned_pdb)
    if not success:
        print(f"  ⚠️  Terminal cleaning failed, using original PDB")
        cleaned_pdb = rna_pdb

    # Generate tleap script
    tleap_script = output_dir / "rna_tleap.in"
    prmtop_file = output_dir / "rna.prmtop"
    inpcrd_file = output_dir / "rna.inpcrd"

    script_content = f"""source leaprc.RNA.OL3
mol = loadpdb {cleaned_pdb.name}

# For pocket fragments, accept structure as-is
set default nocenter on
set default PBRadii mbondi3

# Save topology and coordinates
saveamberparm mol {prmtop_file.name} {inpcrd_file.name}
quit
"""

    tleap_script.write_text(script_content)
    print(f"  Created tleap script")

    # Run tleap
    print(f"  Running tleap...")
    try:
        result = subprocess.run(
            ["tleap", "-f", tleap_script.name],
            capture_output=True,
            text=True,
            cwd=str(output_dir),
            timeout=300
        )

        if prmtop_file.exists() and inpcrd_file.exists():
            print(f"  ✓ Successfully created {prmtop_file.name} and {inpcrd_file.name}")
            return True, prmtop_file, inpcrd_file
        else:
            print(f"  ✗ tleap failed")
            if result.stdout:
                print(f"  stdout: {result.stdout[-500:]}")
            if result.stderr:
                print(f"  stderr: {result.stderr[-500:]}")
            return False, None, None

    except subprocess.TimeoutExpired:
        print(f"  ✗ tleap timeout")
        return False, None, None
    except Exception as e:
        print(f"  ✗ Error running tleap: {e}")
        return False, None, None


def build_graph_from_pocket(rna_pdb: Path, prmtop_path: Path, distance_cutoff: float = 4.0) -> Optional[Data]:
    """
    Build molecular graph from RNA pocket PDB and topology files.

    This function creates node features combining:
    - RDKit features: atomic number, hybridization, aromaticity, degree, formal charge
    - AMBER features: partial charges, atom types

    Args:
        rna_pdb: Path to RNA PDB file
        prmtop_path: Path to AMBER prmtop file
        distance_cutoff: Distance cutoff for edge construction (Angstroms)

    Returns:
        PyTorch Geometric Data object with node features, positions, and edges
    """
    print(f"\n{'─'*70}")
    print(f"Building molecular graph")
    print(f"{'─'*70}")

    try:
        # Load RNA with RDKit
        mol = Chem.MolFromPDBFile(str(rna_pdb), removeHs=False, sanitize=False)
        if mol is None:
            print(f"  ✗ RDKit failed to load {rna_pdb}")
            return None

        # Try to sanitize
        try:
            Chem.SanitizeMol(mol)
        except:
            print(f"  ⚠️  Sanitization failed, continuing anyway")

        # Load AMBER topology
        amber_parm = pmd.load_file(str(prmtop_path))

        # Get number of atoms
        n_atoms = mol.GetNumAtoms()
        n_amber_atoms = len(amber_parm.atoms)

        if n_atoms != n_amber_atoms:
            print(f"  ⚠️  Atom count mismatch - RDKit: {n_atoms}, AMBER: {n_amber_atoms}")
            n_atoms = min(n_atoms, n_amber_atoms)

        print(f"  Processing {n_atoms} atoms...")

        # Extract RDKit features
        rdkit_features = []
        for i in range(n_atoms):
            atom = mol.GetAtomWithIdx(i)

            # Atomic number
            atomic_num = atom.GetAtomicNum()

            # Hybridization (one-hot encoding)
            hybridization = atom.GetHybridization()
            hyb_encoding = [0] * 5  # SP, SP2, SP3, SP3D, SP3D2
            if hybridization == Chem.HybridizationType.SP:
                hyb_encoding[0] = 1
            elif hybridization == Chem.HybridizationType.SP2:
                hyb_encoding[1] = 1
            elif hybridization == Chem.HybridizationType.SP3:
                hyb_encoding[2] = 1
            elif hybridization == Chem.HybridizationType.SP3D:
                hyb_encoding[3] = 1
            elif hybridization == Chem.HybridizationType.SP3D2:
                hyb_encoding[4] = 1

            # Aromaticity
            is_aromatic = float(atom.GetIsAromatic())

            # Degree
            degree = atom.GetDegree()

            # Formal charge
            formal_charge = atom.GetFormalCharge()

            rdkit_features.append([
                atomic_num,
                *hyb_encoding,
                is_aromatic,
                degree,
                formal_charge
            ])

        rdkit_features = np.array(rdkit_features, dtype=np.float32)

        # Extract AMBER features
        amber_features = []
        for i in range(n_atoms):
            atom = amber_parm.atoms[i]

            # Partial charge
            charge = float(atom.charge)

            # AMBER atom type (convert to hash)
            atom_type_hash = hash(atom.type) % 1000

            amber_features.append([charge, atom_type_hash])

        amber_features = np.array(amber_features, dtype=np.float32)

        # Concatenate features
        node_features = np.concatenate([rdkit_features, amber_features], axis=1)
        x = torch.tensor(node_features, dtype=torch.float)

        print(f"  Node features shape: {x.shape}")

        # Extract 3D coordinates
        conf = mol.GetConformer()
        positions = []
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])

        pos = torch.tensor(positions, dtype=torch.float)

        # Build edges based on distance cutoff
        print(f"  Building edges with {distance_cutoff}Å cutoff...")
        edge_index = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                if dist <= distance_cutoff:
                    # Add both directions (undirected graph)
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        if len(edge_index) == 0:
            print(f"  ⚠️  No edges found with cutoff {distance_cutoff}Å, adding self-loops")
            edge_index = [[i, i] for i in range(n_atoms)]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        print(f"  Edges: {edge_index.shape[1]} edges")

        # Create PyTorch Geometric Data object
        data = Data(x=x, pos=pos, edge_index=edge_index)

        print(f"  ✓ Graph construction successful")
        print(f"    - Nodes: {data.x.shape[0]}")
        print(f"    - Node features: {data.x.shape[1]}")
        print(f"    - Edges: {data.edge_index.shape[1]}")

        return data

    except Exception as e:
        print(f"  ✗ Error building graph: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_rna_from_pocket(input_pdb: Path, output_pdb: Path, ligand_resname: Optional[str] = None) -> bool:
    """
    Extract RNA residues from a pocket PDB file (which may contain ligands, proteins, etc.).

    Args:
        input_pdb: Input PDB file (may contain RNA + ligand + protein)
        output_pdb: Output PDB file (RNA only)
        ligand_resname: Optional ligand residue name to exclude

    Returns:
        True if successful
    """
    print(f"\n{'─'*70}")
    print(f"Extracting RNA from pocket")
    print(f"{'─'*70}")

    try:
        u = mda.Universe(str(input_pdb))

        # Select RNA residues
        rna_selection = " or ".join([f"resname {res}" for res in RNA_RESIDUES + MODIFIED_RNA])
        rna_atoms = u.select_atoms(rna_selection)

        # Exclude ligand if specified
        if ligand_resname and len(ligand_resname) > 0:
            rna_atoms = rna_atoms - u.select_atoms(f"resname {ligand_resname}")

        if len(rna_atoms) == 0:
            print(f"  ✗ No RNA atoms found")
            return False

        print(f"  Found {len(rna_atoms)} RNA atoms in {len(rna_atoms.residues)} residues")

        # Save RNA only
        rna_atoms.write(str(output_pdb))
        print(f"  ✓ Saved RNA to {output_pdb}")

        return True

    except Exception as e:
        print(f"  ✗ Error extracting RNA: {e}")
        return False


def generate_graph_from_pocket(input_pdb: Path,
                               output_graph: Path,
                               ligand_resname: Optional[str] = None,
                               distance_cutoff: float = 4.0,
                               keep_intermediate: bool = False) -> bool:
    """
    Main function: Generate PyTorch Geometric graph from RNA binding pocket.

    Args:
        input_pdb: Input PDB file (RNA pocket, may include ligand/protein)
        output_graph: Output graph file (.pt)
        ligand_resname: Optional ligand residue name to exclude from RNA
        distance_cutoff: Distance cutoff for edge construction (Angstroms)
        keep_intermediate: Keep intermediate files (for debugging)

    Returns:
        True if successful
    """
    print(f"\n{'='*70}")
    print(f"Generating graph from RNA binding pocket")
    print(f"{'='*70}")
    print(f"Input: {input_pdb}")
    print(f"Output: {output_graph}")
    if ligand_resname:
        print(f"Excluding ligand: {ligand_resname}")
    print(f"Distance cutoff: {distance_cutoff}Å")
    print(f"{'='*70}\n")

    # Create temporary directory for intermediate files
    temp_dir = Path(tempfile.mkdtemp(prefix="rna_pocket_"))

    try:
        # Step 1: Extract RNA from pocket (if needed)
        rna_pdb = temp_dir / "rna_only.pdb"
        success = extract_rna_from_pocket(input_pdb, rna_pdb, ligand_resname)
        if not success:
            return False

        # Step 2: Parameterize RNA with Amber
        success, prmtop_path, inpcrd_path = parameterize_rna_pocket(rna_pdb, temp_dir)
        if not success:
            return False

        # Step 3: Build graph
        graph_data = build_graph_from_pocket(rna_pdb, prmtop_path, distance_cutoff)
        if graph_data is None:
            return False

        # Step 4: Save graph
        output_graph.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph_data, output_graph)
        print(f"\n{'='*70}")
        print(f"✓ Graph saved to {output_graph}")
        print(f"{'='*70}\n")

        # Clean up or keep intermediate files
        if keep_intermediate:
            intermediate_dir = output_graph.parent / f"{output_graph.stem}_intermediate"
            shutil.copytree(temp_dir, intermediate_dir, dirs_exist_ok=True)
            print(f"Intermediate files saved to {intermediate_dir}")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up temp directory
        if temp_dir.exists() and not keep_intermediate:
            shutil.rmtree(temp_dir)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate PyTorch Geometric graph from RNA binding pocket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - generate graph from RNA pocket PDB
  python generate_pocket_graph.py --input pocket.pdb --output pocket_graph.pt

  # Exclude ligand from RNA selection
  python generate_pocket_graph.py --input pocket.pdb --output pocket_graph.pt --ligand_resname ATP

  # Custom distance cutoff for edges
  python generate_pocket_graph.py --input pocket.pdb --output pocket_graph.pt --distance_cutoff 5.0

  # Keep intermediate files for debugging
  python generate_pocket_graph.py --input pocket.pdb --output pocket_graph.pt --keep_intermediate
"""
    )

    parser.add_argument("--input", type=str, required=True,
                       help="Input PDB file (RNA binding pocket)")
    parser.add_argument("--output", type=str, required=True,
                       help="Output graph file (.pt)")
    parser.add_argument("--ligand_resname", type=str, default=None,
                       help="Ligand residue name to exclude from RNA (optional)")
    parser.add_argument("--distance_cutoff", type=float, default=4.0,
                       help="Distance cutoff for edge construction in Angstroms (default: 4.0)")
    parser.add_argument("--keep_intermediate", action="store_true",
                       help="Keep intermediate files for debugging")

    args = parser.parse_args()

    # Convert to Path objects
    input_pdb = Path(args.input)
    output_graph = Path(args.output)

    # Check input exists
    if not input_pdb.exists():
        print(f"Error: Input file not found: {input_pdb}")
        return 1

    # Generate graph
    success = generate_graph_from_pocket(
        input_pdb,
        output_graph,
        ligand_resname=args.ligand_resname,
        distance_cutoff=args.distance_cutoff,
        keep_intermediate=args.keep_intermediate
    )

    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
