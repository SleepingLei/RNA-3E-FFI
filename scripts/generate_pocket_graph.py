#!/usr/bin/env python3
"""
Generate Graph from Pocket Structure for Virtual Screening

This script takes a pocket structure file (PDB or MOL2 format) as input,
parameterizes it using AMBER force fields, and generates a molecular graph
with the same format as 03_build_dataset.py, suitable for virtual screening.

Key features:
- Node features: AMBER atom types, charges, residues (using amber_vocabulary)
- 1-hop edges: Covalent bonds (without hydrogen)
- 2-hop paths: Angle interactions
- 3-hop paths: Dihedral interactions
- Non-bonded edges: Spatial proximity with LJ parameters

Usage:
    python generate_pocket_graph.py --input pocket.pdb --output pocket_graph.pt
    python generate_pocket_graph.py --input pocket.mol2 --output pocket_graph.pt --molecule_type rna

Input formats supported:
    - PDB: Protein Data Bank format
    - MOL2: Sybyl MOL2 format

Output:
    - PyTorch Geometric graph file (.pt) compatible with 03_build_dataset.py format
"""

import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
import parmed as pmd
import MDAnalysis as mda
import tempfile
import shutil
import sys

# Import AMBER vocabulary utilities
sys.path.append(str(Path(__file__).parent.parent))
from amber_vocabulary import get_global_encoder


def load_normalization_params(norm_params_path):
    """
    Load node feature normalization parameters from .npz file.

    Args:
        norm_params_path: Path to normalization parameters file

    Returns:
        dict: Dictionary containing 'mean', 'std', and 'continuous_indices'
              Returns None if file doesn't exist or loading fails
    """
    if norm_params_path is None:
        return None

    norm_params_path = Path(norm_params_path)
    if not norm_params_path.exists():
        print(f"Warning: Normalization params file not found: {norm_params_path}")
        return None

    try:
        data = np.load(str(norm_params_path))
        params = {
            'mean': data['mean'],
            'std': data['std'],
            'continuous_indices': data['continuous_indices']
        }
        print(f"  Loaded normalization parameters from {norm_params_path.name}")
        print(f"    Mean: {params['mean']}")
        print(f"    Std: {params['std']}")
        print(f"    Continuous indices: {params['continuous_indices']}")
        return params
    except Exception as e:
        print(f"Warning: Failed to load normalization params: {e}")
        return None


# RNA residue types
RNA_RESIDUES = ['A', 'C', 'G', 'U', 'A3', 'A5', 'C3', 'C5', 'G3', 'G5', 'U3', 'U5',
                'DA', 'DC', 'DG', 'DT', 'DA3', 'DA5', 'DC3', 'DC5', 'DG3', 'DG5', 'DT3', 'DT5']

MODIFIED_RNA = ['PSU', '5MU', '5MC', '1MA', '7MG', 'M2G', 'OMC', 'OMG', 'H2U',
                '2MG', 'M7G', 'OMU', 'YYG', 'YG', '6MZ', 'IU', 'I']

PROTEIN_RESIDUES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
                    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
                    'TYR', 'VAL', 'HID', 'HIE', 'HIP', 'CYX']


def detect_molecule_type(input_file):
    """
    Detect the type of molecule in the input file (RNA, protein, or mixed).

    Args:
        input_file: Path to input PDB or MOL2 file

    Returns:
        str: 'rna', 'protein', 'mixed', or 'unknown'
    """
    try:
        u = mda.Universe(str(input_file))

        rna_count = 0
        protein_count = 0

        for residue in u.residues:
            resname = residue.resname.strip()
            if resname in RNA_RESIDUES or resname in MODIFIED_RNA:
                rna_count += 1
            elif resname in PROTEIN_RESIDUES:
                protein_count += 1

        total_residues = len(u.residues)

        if rna_count > 0 and protein_count > 0:
            return 'mixed'
        elif rna_count > total_residues * 0.5:
            return 'rna'
        elif protein_count > total_residues * 0.5:
            return 'protein'
        else:
            return 'unknown'

    except Exception as e:
        print(f"Warning: Could not detect molecule type: {e}")
        return 'unknown'


def clean_rna_terminal_atoms(input_pdb, output_pdb):
    """
    Remove problematic terminal atoms from RNA fragments.
    tleap will add them back with correct types.

    Returns:
        bool: Success status
    """
    print(f"  Cleaning terminal atoms...")

    try:
        u = mda.Universe(str(input_pdb))

        if len(u.residues) == 0:
            print(f"    Warning: No residues found")
            return False

        # Group residues by chain/segment ID
        chains = {}
        for residue in u.residues:
            chain_id = residue.segid if residue.segid else 'X'
            if chain_id not in chains:
                chains[chain_id] = []
            chains[chain_id].append(residue)

        print(f"    Found {len(chains)} chain(s): {list(chains.keys())}")

        # Atoms to remove
        atoms_to_remove = []
        total_removed_5prime = 0
        total_removed_3prime = 0

        # For each chain, clean 5' and 3' terminals
        atoms_to_remove_5prime = {'P', 'OP1', 'OP2', 'O5P', 'O1P', 'O2P'}
        atoms_to_remove_3prime = {"O3'", "O3*"}

        for chain_id, residues in chains.items():
            if len(residues) == 0:
                continue

            # Clean 5' terminal (first residue of this chain)
            first_residue = residues[0]
            removed_5prime = []
            for atom in first_residue.atoms:
                if atom.name in atoms_to_remove_5prime:
                    atoms_to_remove.append(atom)
                    removed_5prime.append(atom.name)
            total_removed_5prime += len(removed_5prime)

            # Clean 3' terminal (last residue of this chain)
            last_residue = residues[-1]
            removed_3prime = []
            for atom in last_residue.atoms:
                if atom.name in atoms_to_remove_3prime:
                    atoms_to_remove.append(atom)
                    removed_3prime.append(atom.name)
            total_removed_3prime += len(removed_3prime)

            if removed_5prime or removed_3prime:
                print(f"      Chain {chain_id}: removed {len(removed_5prime)} 5'-terminal, {len(removed_3prime)} 3'-terminal atoms")

        # Create new AtomGroup without these atoms
        if atoms_to_remove:
            remove_indices = {atom.index for atom in atoms_to_remove}
            keep_atoms = u.atoms[[i for i in range(len(u.atoms)) if i not in remove_indices]]

            # Save cleaned PDB
            keep_atoms.write(str(output_pdb))
            print(f"    Cleaned {len(atoms_to_remove)} terminal atoms ({total_removed_5prime} from 5', {total_removed_3prime} from 3')")
            return True
        else:
            print(f"    No terminal atoms to remove, copying original")
            shutil.copy(input_pdb, output_pdb)
            return True

    except Exception as e:
        print(f"    Error cleaning terminals: {e}")
        return False


def parameterize_pocket(input_file, output_dir, molecule_type='auto'):
    """
    Parameterize pocket structure using AMBER force fields.

    Args:
        input_file: Path to input PDB or MOL2 file
        output_dir: Output directory for AMBER files
        molecule_type: Type of molecule ('rna', 'protein', 'mixed', or 'auto')

    Returns:
        tuple: (success, prmtop_path, inpcrd_path)
    """
    print(f"\n{'─'*70}")
    print(f"Parameterizing pocket structure...")
    print(f"{'─'*70}")

    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect molecule type if needed
    if molecule_type == 'auto':
        molecule_type = detect_molecule_type(input_file)
        print(f"  Auto-detected molecule type: {molecule_type}")

    # Convert to PDB if needed (tleap prefers PDB)
    if input_file.suffix.lower() == '.mol2':
        print(f"  Converting MOL2 to PDB...")
        pdb_file = output_dir / f"{input_file.stem}.pdb"
        try:
            u = mda.Universe(str(input_file))
            u.atoms.write(str(pdb_file))
        except Exception as e:
            print(f"  Error converting MOL2 to PDB: {e}")
            return False, None, None
    else:
        pdb_file = input_file

    # Clean terminal atoms for RNA
    if molecule_type in ['rna', 'mixed']:
        cleaned_pdb = output_dir / f"{pdb_file.stem}_cleaned.pdb"
        success = clean_rna_terminal_atoms(pdb_file, cleaned_pdb)
        if not success:
            print(f"  Warning: Terminal cleaning failed, using original PDB")
            cleaned_pdb = pdb_file
    else:
        cleaned_pdb = pdb_file

    # Generate tleap script based on molecule type
    tleap_script = output_dir / "tleap.in"
    prmtop_file = output_dir / "pocket.prmtop"
    inpcrd_file = output_dir / "pocket.inpcrd"

    # Select appropriate force field
    if molecule_type == 'rna':
        leaprc = "leaprc.RNA.OL3"
    elif molecule_type == 'protein':
        leaprc = "leaprc.protein.ff14SB"
    elif molecule_type == 'mixed':
        # Load both force fields
        leaprc = "leaprc.RNA.OL3\nsource leaprc.protein.ff14SB"
    else:
        # Default to RNA
        print(f"  Warning: Unknown molecule type, defaulting to RNA force field")
        leaprc = "leaprc.RNA.OL3"

    script_content = f"""source {leaprc}
mol = loadpdb {cleaned_pdb.name}

# For pocket fragments, accept structure as-is
set default nocenter on
set default PBRadii mbondi3

# Try to save even with warnings
saveamberparm mol {prmtop_file.name} {inpcrd_file.name}
quit
"""

    tleap_script.write_text(script_content)
    print(f"  Created tleap script for {molecule_type} molecule")

    # Run tleap
    print(f"  Running tleap...")
    result = subprocess.run(
        ["tleap", "-f", tleap_script.name],
        capture_output=True,
        text=True,
        cwd=str(output_dir),
        timeout=300
    )

    if prmtop_file.exists() and inpcrd_file.exists():
        print(f"  Successfully created {prmtop_file.name} and {inpcrd_file.name}")
        # Cleanup
        tleap_script.unlink()
        return True, prmtop_file, inpcrd_file
    else:
        print(f"  Error: tleap failed")
        if result.stdout:
            print(f"  stdout: {result.stdout[-500:]}")
        if result.stderr:
            print(f"  stderr: {result.stderr[-500:]}")
        return False, None, None


def build_graph_from_pocket(prmtop_path, distance_cutoff=5.0, add_nonbonded_edges=True, normalization_params=None):
    """
    Build a molecular graph from AMBER topology files.
    This uses the SAME format as 03_build_dataset.py for compatibility.

    Graph structure:
    - Node features: AMBER atom types, charges, residues, atomic numbers
    - 1-hop edges: Covalent bonds (without hydrogen)
    - 2-hop paths: Angle interactions
    - 3-hop paths: Dihedral interactions
    - Non-bonded edges: Spatial proximity with LJ parameters

    Args:
        prmtop_path: Path to AMBER prmtop file
        distance_cutoff: Distance cutoff for non-bonded edges (Angstroms)
        add_nonbonded_edges: Whether to add non-bonded spatial edges
        normalization_params: Dict with 'mean', 'std', 'continuous_indices' for feature normalization

    Returns:
        torch_geometric.data.Data object with multi-hop graph structure
    """
    try:
        print(f"\n{'─'*70}")
        print(f"Building molecular graph...")
        print(f"{'─'*70}")

        # Load AMBER topology with ParmEd
        amber_parm = pmd.load_file(str(prmtop_path))
        n_atoms = len(amber_parm.atoms)

        if n_atoms == 0:
            print(f"Error: No atoms found in {prmtop_path}")
            return None

        print(f"  Loaded {n_atoms} atoms")

        # ========================================
        # 1. NODE FEATURES (AMBER-based with fixed vocabulary)
        # ========================================

        print(f"  Extracting node features...")
        encoder = get_global_encoder()

        node_features = []
        for atom in amber_parm.atoms:
            features = encoder.encode_atom_features(
                atom_type=atom.type,
                charge=float(atom.charge),
                residue_name=atom.residue.name,
                atomic_number=atom.atomic_number
            )
            node_features.append(features)

        node_features_array = np.array(node_features, dtype=np.float32)

        # Apply normalization if parameters are provided
        if normalization_params is not None:
            print(f"  Applying normalization to node features...")
            mean = normalization_params['mean']
            std = normalization_params['std']
            continuous_indices = normalization_params['continuous_indices']

            # Normalize continuous features (charge and atomic_number)
            for idx in continuous_indices:
                if idx < node_features_array.shape[1]:
                    node_features_array[:, idx] = (node_features_array[:, idx] - mean[idx]) / std[idx]

            print(f"    Normalized features at indices: {continuous_indices}")

        x = torch.from_numpy(node_features_array)
        print(f"    Node features shape: {x.shape}")

        # ========================================
        # 2. NODE POSITIONS (from INPCRD)
        # ========================================

        print(f"  Loading coordinates...")
        inpcrd_path = Path(str(prmtop_path).replace('.prmtop', '.inpcrd'))

        if not inpcrd_path.exists():
            print(f"Error: INPCRD file not found: {inpcrd_path}")
            return None

        try:
            coords = pmd.load_file(str(prmtop_path), str(inpcrd_path))
            positions = coords.coordinates  # [n_atoms, 3] numpy array
            positions = positions.tolist()
        except Exception as e:
            print(f"Error loading INPCRD file {inpcrd_path}: {e}")
            return None

        pos = torch.tensor(positions, dtype=torch.float)
        print(f"    Positions shape: {pos.shape}")

        # ========================================
        # 3. BONDED EDGES (1-hop)
        # ========================================

        print(f"  Building bonded edges (1-hop)...")
        edge_index = []
        edge_attr_bonded = []

        for bond in amber_parm.bonds:
            # Skip bonds involving hydrogen
            atom1, atom2 = bond.atom1, bond.atom2
            if atom1.atomic_number == 1 or atom2.atomic_number == 1:
                continue

            i, j = atom1.idx, atom2.idx

            # Add both directions for undirected graph
            edge_index.append([i, j])
            edge_index.append([j, i])

            # Bond parameters: [equilibrium_length, force_constant]
            bond_params = [bond.type.req if bond.type else 1.5,
                          bond.type.k if bond.type else 300.0]
            edge_attr_bonded.extend([bond_params, bond_params])

        if len(edge_index) == 0:
            print(f"    Warning: No non-hydrogen bonds found")
            edge_index = [[i, i] for i in range(n_atoms)]  # Self-loops fallback
            edge_attr_bonded = [[1.5, 300.0]] * n_atoms

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_bonded = torch.tensor(edge_attr_bonded, dtype=torch.float)

        # Normalize bond attributes
        if edge_attr_bonded.shape[0] > 0:
            req = edge_attr_bonded[:, 0]
            k = edge_attr_bonded[:, 1]
            req_norm = req / 2.0
            k_norm = k / 500.0
            edge_attr_bonded = torch.stack([req_norm, k_norm], dim=1)

        print(f"    Bonded edges: {edge_index.shape[1]}")

        # ========================================
        # 4. ANGLE PATHS (2-hop)
        # ========================================

        print(f"  Building angle paths (2-hop)...")
        triple_index = []
        triple_attr = []

        for angle in amber_parm.angles:
            # Skip angles involving hydrogen
            atom1, atom2, atom3 = angle.atom1, angle.atom2, angle.atom3
            if atom1.atomic_number == 1 or atom2.atomic_number == 1 or atom3.atomic_number == 1:
                continue

            i, j, k = atom1.idx, atom2.idx, atom3.idx

            # Add path: i -> j -> k
            triple_index.append([i, j, k])

            # Angle parameters: [equilibrium_angle, force_constant]
            angle_params = [angle.type.theteq if angle.type else 120.0,
                           angle.type.k if angle.type else 50.0]
            triple_attr.append(angle_params)

        if len(triple_index) > 0:
            triple_index = torch.tensor(triple_index, dtype=torch.long).t().contiguous()
            triple_attr = torch.tensor(triple_attr, dtype=torch.float)

            # Normalize angle attributes
            theta_eq = triple_attr[:, 0]
            k_angle = triple_attr[:, 1]
            theta_eq_norm = theta_eq / 180.0
            k_angle_norm = k_angle / 200.0
            triple_attr = torch.stack([theta_eq_norm, k_angle_norm], dim=1)
        else:
            triple_index = torch.zeros((3, 0), dtype=torch.long)
            triple_attr = torch.zeros((0, 2), dtype=torch.float)

        print(f"    Angle paths: {triple_index.shape[1]}")

        # ========================================
        # 5. DIHEDRAL PATHS (3-hop)
        # ========================================

        print(f"  Building dihedral paths (3-hop)...")
        quadra_index = []
        quadra_attr = []

        for dihedral in amber_parm.dihedrals:
            # Skip dihedrals involving hydrogen
            atom1, atom2, atom3, atom4 = dihedral.atom1, dihedral.atom2, dihedral.atom3, dihedral.atom4
            if any(atom.atomic_number == 1 for atom in [atom1, atom2, atom3, atom4]):
                continue

            i, j, k, l = atom1.idx, atom2.idx, atom3.idx, atom4.idx

            # Add path: i -> j -> k -> l
            quadra_index.append([i, j, k, l])

            # Dihedral parameters
            dihedral_params = [
                dihedral.type.phi_k if dihedral.type else 1.0,
                dihedral.type.per if dihedral.type else 2.0,
                dihedral.type.phase if dihedral.type else 0.0
            ]
            quadra_attr.append(dihedral_params)

        if len(quadra_index) > 0:
            quadra_index = torch.tensor(quadra_index, dtype=torch.long).t().contiguous()
            quadra_attr = torch.tensor(quadra_attr, dtype=torch.float)

            # Normalize dihedral attributes
            phi_k = quadra_attr[:, 0]
            per = quadra_attr[:, 1]
            phase = quadra_attr[:, 2]
            phi_k_norm = phi_k / 20.0
            per_norm = per / 6.0
            phase_norm = phase / (2 * 3.14159)
            quadra_attr = torch.stack([phi_k_norm, per_norm, phase_norm], dim=1)
        else:
            quadra_index = torch.zeros((4, 0), dtype=torch.long)
            quadra_attr = torch.zeros((0, 3), dtype=torch.float)

        print(f"    Dihedral paths: {quadra_index.shape[1]}")

        # ========================================
        # 6. NON-BONDED EDGES (optional)
        # ========================================

        print(f"  Building non-bonded edges...")
        nonbonded_edges = []
        nonbonded_attr = []

        if add_nonbonded_edges:
            # Build set of bonded pairs to exclude
            bonded_pairs = set()
            for bond in amber_parm.bonds:
                i, j = bond.atom1.idx, bond.atom2.idx
                bonded_pairs.add((min(i, j), max(i, j)))

            # Extract LJ parameters
            try:
                lj_acoef = np.array(amber_parm.parm_data['LENNARD_JONES_ACOEF'], dtype=np.float64)
                lj_bcoef = np.array(amber_parm.parm_data['LENNARD_JONES_BCOEF'], dtype=np.float64)
                nb_parm_index = np.array(amber_parm.parm_data['NONBONDED_PARM_INDEX'], dtype=np.int32)
                ntypes = amber_parm.ptr('ntypes')
            except KeyError as e:
                print(f"    Warning: Could not extract LJ parameters: {e}")
                lj_acoef = None
                lj_bcoef = None

            positions_array = np.array(positions)

            # Add spatial edges within cutoff
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    if (i, j) in bonded_pairs:
                        continue

                    dist = np.linalg.norm(positions_array[i] - positions_array[j])
                    if dist <= distance_cutoff:
                        # Extract LJ parameters
                        if lj_acoef is not None and lj_bcoef is not None:
                            type_i = amber_parm.atoms[i].nb_idx - 1
                            type_j = amber_parm.atoms[j].nb_idx - 1
                            parm_idx = nb_parm_index[type_i * ntypes + type_j] - 1
                            lj_A = float(lj_acoef[parm_idx])
                            lj_B = float(lj_bcoef[parm_idx])
                        else:
                            lj_A = 0.0
                            lj_B = 0.0

                        # Add both directions
                        nonbonded_edges.append([i, j])
                        nonbonded_edges.append([j, i])
                        nonbonded_attr.extend([[lj_A, lj_B, dist], [lj_A, lj_B, dist]])

        if len(nonbonded_edges) > 0:
            nonbonded_edge_index = torch.tensor(nonbonded_edges, dtype=torch.long).t().contiguous()
            nonbonded_edge_attr = torch.tensor(nonbonded_attr, dtype=torch.float)

            # Normalize with log transformation
            lj_a = nonbonded_edge_attr[:, 0]
            lj_b = nonbonded_edge_attr[:, 1]
            dist = nonbonded_edge_attr[:, 2]
            lj_a_log = torch.log(lj_a + 1.0)
            lj_b_log = torch.log(lj_b + 1.0)
            nonbonded_edge_attr = torch.stack([lj_a_log, lj_b_log, dist], dim=1)
        else:
            nonbonded_edge_index = torch.zeros((2, 0), dtype=torch.long)
            nonbonded_edge_attr = torch.zeros((0, 3), dtype=torch.float)

        print(f"    Non-bonded edges: {nonbonded_edge_index.shape[1]}")

        # ========================================
        # 7. CREATE DATA OBJECT
        # ========================================

        data = Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr_bonded,
            triple_index=triple_index,
            triple_attr=triple_attr,
            quadra_index=quadra_index,
            quadra_attr=quadra_attr,
            nonbonded_edge_index=nonbonded_edge_index,
            nonbonded_edge_attr=nonbonded_edge_attr
        )

        print(f"\n  Graph summary:")
        print(f"    Nodes: {data.x.shape[0]}")
        print(f"    Node features: {data.x.shape[1]}")
        print(f"    Bonded edges: {data.edge_index.shape[1]}")
        print(f"    Angle paths: {data.triple_index.shape[1]}")
        print(f"    Dihedral paths: {data.quadra_index.shape[1]}")
        print(f"    Non-bonded edges: {data.nonbonded_edge_index.shape[1]}")

        return data

    except Exception as e:
        print(f"Error building graph: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main pipeline for generating graph from pocket structure."""
    parser = argparse.ArgumentParser(
        description="Generate molecular graph from pocket structure for virtual screening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate graph from pocket PDB (auto-detect molecule type)
    python generate_pocket_graph.py --input pocket.pdb --output pocket_graph.pt

    # Generate graph with normalization (recommended for inference)
    python generate_pocket_graph.py --input pocket.pdb --output pocket_graph.pt \\
        --norm_params ../data/processed/node_feature_norm_params.npz

    # Generate graph from MOL2 file with custom distance cutoff
    python generate_pocket_graph.py --input pocket.mol2 --output output_graph.pt --distance_cutoff 6.0

    # Specify molecule type explicitly (rna, protein, or mixed)
    python generate_pocket_graph.py --input pocket.pdb --output graph.pt --molecule_type rna

    # Keep temporary AMBER files for debugging
    python generate_pocket_graph.py --input pocket.pdb --output graph.pt --keep_temp
        """
    )

    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input pocket structure file (PDB or MOL2 format)")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output graph file path (.pt format)")
    parser.add_argument("--distance_cutoff", type=float, default=5.0,
                       help="Distance cutoff for non-bonded edges (Angstroms, default: 5.0)")
    parser.add_argument("--molecule_type", type=str, default='auto',
                       choices=['auto', 'rna', 'protein', 'mixed'],
                       help="Type of molecule in pocket (default: auto-detect)")
    parser.add_argument("--no_nonbonded_edges", action="store_true",
                       help="Skip non-bonded edge construction")
    parser.add_argument("--norm_params", type=str, default=None,
                       help="Path to node feature normalization parameters (.npz file)")
    parser.add_argument("--keep_temp", action="store_true",
                       help="Keep temporary AMBER files for debugging")

    args = parser.parse_args()

    # Validate input
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return 1

    if input_file.suffix.lower() not in ['.pdb', '.mol2']:
        print(f"Error: Unsupported input format: {input_file.suffix}")
        print(f"       Supported formats: .pdb, .mol2")
        return 1

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Pocket Graph Generation Pipeline")
    print(f"{'='*70}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Distance cutoff: {args.distance_cutoff} Å")
    print(f"Molecule type: {args.molecule_type}")
    print(f"{'='*70}\n")

    # Create temporary directory for AMBER files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Step 1: Parameterize pocket
        success, prmtop_path, inpcrd_path = parameterize_pocket(
            input_file,
            temp_dir,
            molecule_type=args.molecule_type
        )

        if not success:
            print(f"\nError: Parameterization failed")
            return 1

        # Step 1.5: Load normalization parameters
        normalization_params = None
        if args.norm_params:
            normalization_params = load_normalization_params(args.norm_params)
            if normalization_params is None:
                print(f"Warning: Will proceed without normalization")

        # Step 2: Build graph
        add_nonbonded = not args.no_nonbonded_edges
        graph_data = build_graph_from_pocket(
            prmtop_path,
            distance_cutoff=args.distance_cutoff,
            add_nonbonded_edges=add_nonbonded,
            normalization_params=normalization_params
        )

        if graph_data is None:
            print(f"\nError: Graph construction failed")
            return 1

        # Step 3: Save graph
        print(f"\n{'─'*70}")
        print(f"Saving graph to {output_file}...")
        torch.save(graph_data, output_file)
        print(f"Graph saved successfully!")

        # Optionally keep temporary files
        if args.keep_temp:
            debug_dir = output_file.parent / f"{output_file.stem}_amber_files"
            debug_dir.mkdir(exist_ok=True)
            shutil.copytree(temp_dir, debug_dir, dirs_exist_ok=True)
            print(f"AMBER files saved to: {debug_dir}")

    print(f"\n{'='*70}")
    print(f"Pipeline completed successfully!")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    exit(main())
