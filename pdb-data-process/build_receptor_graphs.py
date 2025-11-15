#!/usr/bin/env python3
"""
Build molecular graphs from AMBER-parameterized receptor files.
Based on scripts/03_build_dataset.py with multi-hop interactions.
"""
import os
from pathlib import Path
import argparse
import numpy as np
import torch
from torch_geometric.data import Data
import parmed as pmd
from multiprocessing import Pool, cpu_count
import pickle


def build_graph_from_files(prmtop_path, inpcrd_path, distance_cutoff=5.0):
    """Build molecular graph from AMBER topology and coordinate files."""
    try:
        # Load AMBER topology
        amber_parm = pmd.load_file(str(prmtop_path), str(inpcrd_path))
        n_atoms = len(amber_parm.atoms)

        if n_atoms == 0:
            return None

        # Node features: [charge, atomic_number, mass]
        node_features = []
        for atom in amber_parm.atoms:
            charge = float(atom.charge)
            atomic_number = int(atom.atomic_number)
            mass = float(atom.mass)
            node_features.append([charge, atomic_number, mass])

        x = torch.tensor(node_features, dtype=torch.float)

        # Node positions
        positions = amber_parm.coordinates
        pos = torch.tensor(positions, dtype=torch.float)

        # 1-hop edges (bonds without hydrogen)
        edge_index = []
        edge_attr_bonded = []

        for bond in amber_parm.bonds:
            atom1, atom2 = bond.atom1, bond.atom2
            if atom1.atomic_number == 1 or atom2.atomic_number == 1:
                continue

            i, j = atom1.idx, atom2.idx
            edge_index.append([i, j])
            edge_index.append([j, i])

            bond_params = [
                bond.type.req if bond.type else 1.5,
                bond.type.k if bond.type else 300.0
            ]
            edge_attr_bonded.extend([bond_params, bond_params])

        if len(edge_index) == 0:
            edge_index = [[i, i] for i in range(n_atoms)]
            edge_attr_bonded = [[1.5, 300.0]] * n_atoms

        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr_bonded = torch.tensor(edge_attr_bonded, dtype=torch.float)

        # Normalize bond attributes
        if edge_attr_bonded.shape[0] > 0:
            req = edge_attr_bonded[:, 0] / 2.0
            k = edge_attr_bonded[:, 1] / 500.0
            edge_attr_bonded = torch.stack([req, k], dim=1)

        # 2-hop paths (angles without hydrogen)
        triple_index = []
        triple_attr = []

        for angle in amber_parm.angles:
            atom1, atom2, atom3 = angle.atom1, angle.atom2, angle.atom3
            if atom1.atomic_number == 1 or atom2.atomic_number == 1 or atom3.atomic_number == 1:
                continue

            i, j, k = atom1.idx, atom2.idx, atom3.idx
            triple_index.append([i, j, k])

            angle_params = [
                angle.type.theteq if angle.type else 120.0,
                angle.type.k if angle.type else 50.0
            ]
            triple_attr.append(angle_params)

        if len(triple_index) > 0:
            triple_index = torch.tensor(triple_index, dtype=torch.long).t()
            triple_attr = torch.tensor(triple_attr, dtype=torch.float)
            # Normalize
            theta_eq = triple_attr[:, 0] / 180.0
            k_angle = triple_attr[:, 1] / 100.0
            triple_attr = torch.stack([theta_eq, k_angle], dim=1)
        else:
            triple_index = torch.zeros((3, 0), dtype=torch.long)
            triple_attr = torch.zeros((0, 2), dtype=torch.float)

        # 3-hop paths (dihedrals without hydrogen)
        quadra_index = []
        quadra_attr = []

        for dihedral in amber_parm.dihedrals:
            atom1, atom2, atom3, atom4 = dihedral.atom1, dihedral.atom2, dihedral.atom3, dihedral.atom4
            if any(atom.atomic_number == 1 for atom in [atom1, atom2, atom3, atom4]):
                continue

            i, j, k, l = atom1.idx, atom2.idx, atom3.idx, atom4.idx
            quadra_index.append([i, j, k, l])

            dihedral_params = [
                dihedral.type.phi_k if dihedral.type else 1.0,
                dihedral.type.per if dihedral.type else 2.0,
                dihedral.type.phase if dihedral.type else 0.0
            ]
            quadra_attr.append(dihedral_params)

        if len(quadra_index) > 0:
            quadra_index = torch.tensor(quadra_index, dtype=torch.long).t()
            quadra_attr = torch.tensor(quadra_attr, dtype=torch.float)
            # Normalize
            phi_k = quadra_attr[:, 0] / 10.0
            per = quadra_attr[:, 1] / 6.0
            phase = quadra_attr[:, 2] / 180.0
            quadra_attr = torch.stack([phi_k, per, phase], dim=1)
        else:
            quadra_index = torch.zeros((4, 0), dtype=torch.long)
            quadra_attr = torch.zeros((0, 3), dtype=torch.float)

        # Non-bonded edges
        nonbonded_edges = []
        nonbonded_attr = []

        bonded_pairs = set()
        for bond in amber_parm.bonds:
            i, j = bond.atom1.idx, bond.atom2.idx
            bonded_pairs.add((min(i, j), max(i, j)))

        try:
            lj_acoef = np.array(amber_parm.parm_data['LENNARD_JONES_ACOEF'])
            lj_bcoef = np.array(amber_parm.parm_data['LENNARD_JONES_BCOEF'])
            nb_parm_index = np.array(amber_parm.parm_data['NONBONDED_PARM_INDEX'])
            ntypes = amber_parm.ptr('ntypes')
        except:
            lj_acoef = None
            lj_bcoef = None

        positions_array = np.array(positions)

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if (i, j) in bonded_pairs:
                    continue

                dist = np.linalg.norm(positions_array[i] - positions_array[j])
                if dist <= distance_cutoff:
                    if lj_acoef is not None and lj_bcoef is not None:
                        type_i = amber_parm.atoms[i].nb_idx - 1
                        type_j = amber_parm.atoms[j].nb_idx - 1
                        parm_idx = nb_parm_index[type_i * ntypes + type_j] - 1
                        lj_A = float(lj_acoef[parm_idx])
                        lj_B = float(lj_bcoef[parm_idx])
                    else:
                        lj_A = 0.0
                        lj_B = 0.0

                    nonbonded_edges.append([i, j])
                    nonbonded_edges.append([j, i])
                    nonbonded_attr.extend([[lj_A, lj_B, dist], [lj_A, lj_B, dist]])

        if len(nonbonded_edges) > 0:
            nonbonded_edge_index = torch.tensor(nonbonded_edges, dtype=torch.long).t()
            nonbonded_edge_attr = torch.tensor(nonbonded_attr, dtype=torch.float)
            # Normalize
            lj_a_log = torch.log(nonbonded_edge_attr[:, 0] + 1.0)
            lj_b_log = torch.log(nonbonded_edge_attr[:, 1] + 1.0)
            dist = nonbonded_edge_attr[:, 2]
            nonbonded_edge_attr = torch.stack([lj_a_log, lj_b_log, dist], dim=1)
        else:
            nonbonded_edge_index = torch.zeros((2, 0), dtype=torch.long)
            nonbonded_edge_attr = torch.zeros((0, 3), dtype=torch.float)

        # Create PyG Data object
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

        return data

    except Exception as e:
        print(f"Error building graph: {e}")
        return None


def process_single_file(args):
    """Process a single receptor file."""
    pdb_id, prmtop_path, inpcrd_path, output_dir = args

    try:
        graph = build_graph_from_files(prmtop_path, inpcrd_path)

        if graph is None:
            return {'pdb_id': pdb_id, 'status': 'failed'}

        # Save graph
        output_file = output_dir / "graphs" / f"{pdb_id}.pt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, output_file)

        return {
            'pdb_id': pdb_id,
            'status': 'success',
            'graph_file': str(output_file),
            'num_atoms': graph.x.shape[0],
            'num_edges': graph.edge_index.shape[1]
        }

    except Exception as e:
        return {'pdb_id': pdb_id, 'status': f'error: {str(e)}'}


def main():
    parser = argparse.ArgumentParser(description='Build receptor graphs from AMBER files')
    parser.add_argument('--amber-dir', default='effect_receptor_parameterized/amber',
                        help='Directory containing AMBER files')
    parser.add_argument('--output-dir', default='effect_receptor_parameterized',
                        help='Output directory')
    parser.add_argument('--distance-cutoff', type=float, default=5.0,
                        help='Distance cutoff for non-bonded edges')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes')

    args = parser.parse_args()

    amber_dir = Path(args.amber_dir)
    output_dir = Path(args.output_dir)

    # Find all prmtop files
    prmtop_files = list(amber_dir.glob('*.prmtop'))

    if not prmtop_files:
        print(f"No prmtop files found in {amber_dir}")
        return

    print(f"Found {len(prmtop_files)} prmtop files")

    # Prepare arguments
    process_args = []
    for prmtop_file in prmtop_files:
        pdb_id = prmtop_file.stem
        inpcrd_file = prmtop_file.with_suffix('.inpcrd')

        if not inpcrd_file.exists():
            print(f"Warning: Missing inpcrd for {pdb_id}")
            continue

        process_args.append((pdb_id, prmtop_file, inpcrd_file, output_dir))

    if not process_args:
        print("No valid prmtop/inpcrd pairs found!")
        return

    num_workers = args.workers or cpu_count()
    print(f"Processing with {num_workers} workers...")

    with Pool(num_workers) as pool:
        results = pool.map(process_single_file, process_args)

    # Statistics
    success_count = sum(1 for r in results if r['status'] == 'success')

    print(f"\n{'='*70}")
    print(f"GRAPH CONSTRUCTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total: {len(results)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(results) - success_count}")

    # Save summary
    summary_file = output_dir / "graph_summary.pkl"
    with open(summary_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSummary saved to {summary_file}")


if __name__ == '__main__':
    main()
