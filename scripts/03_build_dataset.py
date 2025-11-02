#!/usr/bin/env python3
"""
Graph Construction and Dataset Creation

This script builds molecular graphs from processed RNA structures (RNA-only,
NOT including ligands) and creates PyTorch Geometric datasets for training.

The graphs are built from:
  - RNA PDB files: {pdb_id}_{ligand_name}_model{N}_rna.pdb
  - RNA topology files: {pdb_id}_{ligand_name}_model{N}_rna.prmtop
"""
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem import AllChem
import parmed as pmd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import glob
import ast

# Import AMBER vocabulary utilities
from amber_vocabulary import get_global_encoder


def build_graph_from_files(rna_pdb_path, prmtop_path, distance_cutoff=5.0, add_nonbonded_edges=True):
    """
    Build a molecular graph from RNA PDB and AMBER topology files with FFiNet-style multi-hop interactions.

    This function creates a graph with:
    - Node features: AMBER_ATOM_TYPE, CHARGE, RESIDUE_LABEL
    - 1-hop edges: BONDS_WITHOUT_HYDROGEN (covalent bonds)
    - 2-hop paths: ANGLES_WITHOUT_HYDROGEN (angle interactions)
    - 3-hop paths: DIHEDRALS_WITHOUT_HYDROGEN (torsion interactions)
    - Non-bonded edges: Spatial proximity with LJ parameters

    Args:
        rna_pdb_path: Path to RNA PDB file
        prmtop_path: Path to AMBER prmtop file
        distance_cutoff: Distance cutoff for non-bonded edge construction (Angstroms)
        add_nonbonded_edges: Whether to add non-bonded spatial edges

    Returns:
        torch_geometric.data.Data object with multi-hop graph structure
    """
    try:
        # Load AMBER topology with ParmEd (primary source of truth)
        amber_parm = pmd.load_file(str(prmtop_path))
        n_atoms = len(amber_parm.atoms)

        if n_atoms == 0:
            print(f"Error: No atoms found in {prmtop_path}")
            return None

        # ========================================
        # 1. NODE FEATURES (AMBER-based with fixed vocabulary)
        # ========================================

        # Use global encoder with fixed vocabularies
        encoder = get_global_encoder()

        node_features = []
        for atom in amber_parm.atoms:
            # Encode all features using fixed vocabulary
            features = encoder.encode_atom_features(
                atom_type=atom.type,
                charge=float(atom.charge),
                residue_name=atom.residue.name,
                atomic_number=atom.atomic_number
            )
            node_features.append(features)

        # Convert to numpy first (avoids warning)
        node_features_array = np.array(node_features, dtype=np.float32)
        x = torch.from_numpy(node_features_array)

        # ========================================
        # 2. NODE POSITIONS (only from INPCRD)
        # ========================================

        # Load from .inpcrd file (more reliable than PDB for atom ordering)
        inpcrd_path = Path(str(prmtop_path).replace('.prmtop', '.inpcrd'))
        if not inpcrd_path.exists():
            # Try alternative location
            inpcrd_path = Path(str(rna_pdb_path).parent) / 'rna.inpcrd'

        if not inpcrd_path.exists():
            print(f"Error: INPCRD file not found. Searched: {inpcrd_path}")
            return None

        try:
            # Load coordinates using ParmEd
            coords = pmd.load_file(str(prmtop_path), str(inpcrd_path))
            positions = coords.coordinates  # [n_atoms, 3] numpy array
            positions = positions.tolist()
        except Exception as e:
            print(f"Error loading INPCRD file {inpcrd_path}: {e}")
            return None

        pos = torch.tensor(positions, dtype=torch.float)

        # ========================================
        # 3. BONDED EDGES (1-hop) - BONDS_WITHOUT_HYDROGEN
        # ========================================

        edge_index = []
        edge_attr_bonded = []  # Store bond parameters

        for bond in amber_parm.bonds:
            # Skip bonds involving hydrogen (to match BONDS_WITHOUT_HYDROGEN)
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
            edge_attr_bonded.extend([bond_params, bond_params])  # Same for both directions

        if len(edge_index) == 0:
            print(f"Warning: No non-hydrogen bonds found in {prmtop_path}")
            edge_index = [[i, i] for i in range(n_atoms)]  # Self-loops fallback
            edge_attr_bonded = [[1.5, 300.0]] * n_atoms

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_bonded = torch.tensor(edge_attr_bonded, dtype=torch.float)
        if edge_attr_bonded.shape[0] > 0:
            req = edge_attr_bonded[:, 0]
            k = edge_attr_bonded[:, 1]
            req_norm = req / 2.0
            k_norm = k / 500.0 
            edge_attr_bonded = torch.stack([req_norm, k_norm], dim=1)

        # ========================================
        # 4. ANGLE PATHS (2-hop) - ANGLES_WITHOUT_HYDROGEN
        # ========================================

        triple_index = []  # Format: [src, mid, dst]
        triple_attr = []   # Angle parameters

        for angle in amber_parm.angles:
            # Skip angles involving hydrogen
            atom1, atom2, atom3 = angle.atom1, angle.atom2, angle.atom3
            if atom1.atomic_number == 1 or atom2.atomic_number == 1 or atom3.atomic_number == 1:
                continue

            i, j, k = atom1.idx, atom2.idx, atom3.idx

            # Add path: i -> j -> k (j is the central atom)
            triple_index.append([i, j, k])

            # Angle parameters: [equilibrium_angle, force_constant]
            angle_params = [angle.type.theteq if angle.type else 120.0,
                           angle.type.k if angle.type else 50.0]
            triple_attr.append(angle_params)

        if len(triple_index) > 0:
            triple_index = torch.tensor(triple_index, dtype=torch.long).t().contiguous()
            triple_attr = torch.tensor(triple_attr, dtype=torch.float)
        else:
            triple_index = torch.zeros((3, 0), dtype=torch.long)
            triple_attr = torch.zeros((0, 2), dtype=torch.float)

        # ========================================
        # 5. DIHEDRAL PATHS (3-hop) - DIHEDRALS_WITHOUT_HYDROGEN
        # ========================================

        quadra_index = []  # Format: [src, mid2, mid1, dst]
        quadra_attr = []   # Dihedral parameters

        for dihedral in amber_parm.dihedrals:
            # Skip dihedrals involving hydrogen
            atom1, atom2, atom3, atom4 = dihedral.atom1, dihedral.atom2, dihedral.atom3, dihedral.atom4
            if any(atom.atomic_number == 1 for atom in [atom1, atom2, atom3, atom4]):
                continue

            i, j, k, l = atom1.idx, atom2.idx, atom3.idx, atom4.idx

            # Add path: i -> j -> k -> l
            quadra_index.append([i, j, k, l])

            # Dihedral parameters: [force_constant, periodicity, phase]
            dihedral_params = [
                dihedral.type.phi_k if dihedral.type else 1.0,
                dihedral.type.per if dihedral.type else 2.0,
                dihedral.type.phase if dihedral.type else 0.0
            ]
            quadra_attr.append(dihedral_params)

        if len(quadra_index) > 0:
            quadra_index = torch.tensor(quadra_index, dtype=torch.long).t().contiguous()
            quadra_attr = torch.tensor(quadra_attr, dtype=torch.float)
        else:
            quadra_index = torch.zeros((4, 0), dtype=torch.long)
            quadra_attr = torch.zeros((0, 3), dtype=torch.float)

        # ========================================
        # 6. NON-BONDED EDGES (optional spatial proximity) with real LJ parameters
        # ========================================

        nonbonded_edges = []
        nonbonded_attr = []  # LJ parameters

        if add_nonbonded_edges:
            # Build set of bonded pairs to exclude
            bonded_pairs = set()
            for bond in amber_parm.bonds:
                i, j = bond.atom1.idx, bond.atom2.idx
                bonded_pairs.add((min(i, j), max(i, j)))

            # Extract LJ parameters from prmtop
            try:
                lj_acoef = np.array(amber_parm.parm_data['LENNARD_JONES_ACOEF'], dtype=np.float64)
                lj_bcoef = np.array(amber_parm.parm_data['LENNARD_JONES_BCOEF'], dtype=np.float64)
                nb_parm_index = np.array(amber_parm.parm_data['NONBONDED_PARM_INDEX'], dtype=np.int32)
                ntypes = amber_parm.ptr('ntypes')  # Number of atom types
            except KeyError as e:
                print(f"Warning: Could not extract LJ parameters from prmtop: {e}")
                lj_acoef = None
                lj_bcoef = None

            # Convert positions to numpy array if not already
            positions_array = np.array(positions) if not isinstance(positions, np.ndarray) else positions

            # Add spatial edges within cutoff (excluding bonded)
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    # Skip if already bonded
                    if (i, j) in bonded_pairs:
                        continue

                    # Check distance
                    dist = np.linalg.norm(positions_array[i] - positions_array[j])
                    if dist <= distance_cutoff:
                        # Extract real LJ parameters
                        if lj_acoef is not None and lj_bcoef is not None:
                            # Get atom type indices (nb_idx is 1-indexed in AMBER)
                            type_i = amber_parm.atoms[i].nb_idx - 1
                            type_j = amber_parm.atoms[j].nb_idx - 1

                            # Calculate index into LJ parameter arrays
                            # AMBER uses triangular matrix storage
                            # Index formula: index = nb_parm_index[type_i * ntypes + type_j] - 1
                            parm_idx = nb_parm_index[type_i * ntypes + type_j] - 1

                            # Extract A and B coefficients
                            lj_A = float(lj_acoef[parm_idx])
                            lj_B = float(lj_bcoef[parm_idx])
                        else:
                            # Fallback to placeholder if extraction failed
                            lj_A = 0.0
                            lj_B = 0.0

                        # Add both directions
                        nonbonded_edges.append([i, j])
                        nonbonded_edges.append([j, i])
                        nonbonded_attr.extend([[lj_A, lj_B, dist], [lj_A, lj_B, dist]])

        if len(nonbonded_edges) > 0:
            nonbonded_edge_index = torch.tensor(nonbonded_edges, dtype=torch.long).t().contiguous()
            nonbonded_edge_attr = torch.tensor(nonbonded_attr, dtype=torch.float)

            # Normalize nonbonded attributes to prevent numerical overflow
            # nonbonded_edge_attr: [num_edges, 3] = [LJ_A, LJ_B, distance]
            # Use log transformation for LJ parameters (they can be extremely large: >10^6)
            lj_a = nonbonded_edge_attr[:, 0]
            lj_b = nonbonded_edge_attr[:, 1]
            dist = nonbonded_edge_attr[:, 2]

            # Log transform (add 1 to handle zeros)
            lj_a_log = torch.log(lj_a + 1.0)
            lj_b_log = torch.log(lj_b + 1.0)

            # Reconstruct with normalized values
            nonbonded_edge_attr = torch.stack([lj_a_log, lj_b_log, dist], dim=1)
        else:
            nonbonded_edge_index = torch.zeros((2, 0), dtype=torch.long)
            nonbonded_edge_attr = torch.zeros((0, 3), dtype=torch.float)

        # ========================================
        # Normalize angle and dihedral attributes
        # ========================================
        # Angle attributes: [theta_eq, k] - k (force constant) can be very large
        if triple_attr.shape[0] > 0:
            theta_eq = triple_attr[:, 0]  # Equilibrium angle (degrees)
            k_angle = triple_attr[:, 1]   # Force constant (can be 40-140)

            # Normalize to [0, 1] range approximately
            theta_eq_norm = theta_eq / 180.0  # Degrees to [0, 1]
            k_angle_norm = k_angle / 200.0    # Typical max ~140, normalized

            triple_attr = torch.stack([theta_eq_norm, k_angle_norm], dim=1)

        # Dihedral attributes: [phi_k, per, phase] - phi_k (barrier height) can be large
        if quadra_attr.shape[0] > 0:
            phi_k = quadra_attr[:, 0]   # Barrier height (can be 0-20)
            per = quadra_attr[:, 1]     # Periodicity (1-6)
            phase = quadra_attr[:, 2]   # Phase angle (radians, 0-2π)

            # Normalize
            phi_k_norm = phi_k / 20.0   # Typical max ~20
            per_norm = per / 6.0        # Max periodicity is 6
            phase_norm = phase / (2 * 3.14159)  # Radians to [0, 1]

            quadra_attr = torch.stack([phi_k_norm, per_norm, phase_norm], dim=1)

        # ========================================
        # 7. CREATE DATA OBJECT
        # ========================================

        data = Data(
            x=x,                              # Node features: [num_atoms, feature_dim]
            pos=pos,                          # Positions: [num_atoms, 3]
            edge_index=edge_index,            # Bonded edges (1-hop): [2, num_bonds]
            edge_attr=edge_attr_bonded,       # Bond parameters: [num_bonds, 2]
            triple_index=triple_index,        # Angle paths (2-hop): [3, num_angles]
            triple_attr=triple_attr,          # Angle parameters: [num_angles, 2]
            quadra_index=quadra_index,        # Dihedral paths (3-hop): [4, num_dihedrals]
            quadra_attr=quadra_attr,          # Dihedral parameters: [num_dihedrals, 3]
            nonbonded_edge_index=nonbonded_edge_index,  # Non-bonded edges: [2, num_nonbonded]
            nonbonded_edge_attr=nonbonded_edge_attr     # LJ parameters: [num_nonbonded, 3]
        )

        return data

    except Exception as e:
        print(f"Error building graph from {rna_pdb_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_single_complex(row, pdb_id_column, ligand_column, amber_dir, output_dir, distance_cutoff):
    """
    Process a single complex and all its models.

    Args:
        row: DataFrame row containing complex information
        pdb_id_column: Name of the PDB ID column
        ligand_column: Name of the ligand column
        amber_dir: Path to AMBER files directory
        output_dir: Path to output directory for graphs
        distance_cutoff: Distance cutoff for edge construction

    Returns:
        Tuple of (complex_id, success_count, failed_list)
    """
    try:
        pdb_id = str(row[pdb_id_column]).lower()

        # Parse ligand name from sm_ligand_ids if needed
        ligand_str = str(row[ligand_column])
        if ligand_column == 'sm_ligand_ids':
            # Parse format like "['ARG_.:B/1:N']" or "ARG_.:B/1:N"
            try:
                ligands = ast.literal_eval(ligand_str)
                if not isinstance(ligands, list):
                    ligands = [ligand_str]
            except:
                ligands = [ligand_str]

            if ligands and len(ligands) > 0:
                # Extract "ARG" from "ARG_.:B/1:N"
                ligand_resname = ligands[0].split('_')[0].split(':')[0]
            else:
                ligand_resname = 'LIG'
        else:
            ligand_resname = ligand_str

        complex_id = f"{pdb_id}_{ligand_resname}"

        # Find all model files for this complex (format: {pdb_id}_{ligand}_model{N}_rna.pdb)
        pattern = str(amber_dir / f"{complex_id}_model*_rna.pdb")
        model_pdb_files = sorted(glob.glob(pattern))

        if not model_pdb_files:
            # Fallback: try without model number (for backward compatibility)
            rna_pdb_path = amber_dir / f"{complex_id}_rna.pdb"
            rna_prmtop_path = amber_dir / f"{complex_id}_rna.prmtop"

            if rna_pdb_path.exists() and rna_prmtop_path.exists():
                model_pdb_files = [rna_pdb_path]
            else:
                return (complex_id, 0, [(complex_id, "rna_pdb_not_found (searched both formats)")])

        success_count = 0
        failed_list = []

        # Process each model
        for rna_pdb_path in model_pdb_files:
            rna_pdb_path = Path(rna_pdb_path)

            # Extract model number from filename
            # Format: {pdb_id}_{ligand}_model{N}_rna.pdb
            stem = rna_pdb_path.stem  # e.g., "1aju_ARG_model0_rna"
            if "_model" in stem:
                model_part = stem.split("_model")[1].split("_")[0]  # e.g., "0"
                complex_model_id = f"{complex_id}_model{model_part}"
            else:
                complex_model_id = complex_id

            # Check if graph already exists
            graph_path = output_dir / f"{complex_model_id}.pt"
            if graph_path.exists():
                success_count += 1
                continue

            # Find corresponding prmtop file
            rna_prmtop_path = rna_pdb_path.parent / rna_pdb_path.name.replace("_rna.pdb", "_rna.prmtop")

            # Check if prmtop file exists
            if not rna_prmtop_path.exists():
                failed_list.append((complex_model_id, "rna_prmtop_not_found"))
                continue

            # Check if prmtop file is empty or too small
            prmtop_size = rna_prmtop_path.stat().st_size
            if prmtop_size == 0:
                failed_list.append((complex_model_id, "rna_prmtop_empty (0 bytes)"))
                continue
            elif prmtop_size < 100:  # Suspiciously small
                failed_list.append((complex_model_id, f"rna_prmtop_too_small ({prmtop_size} bytes)"))
                continue

            # Build graph
            try:
                data = build_graph_from_files(rna_pdb_path, rna_prmtop_path, distance_cutoff)

                if data is not None:
                    # Save graph
                    torch.save(data, graph_path)
                    success_count += 1
                else:
                    failed_list.append((complex_model_id, "graph_construction_failed"))

            except Exception as e:
                failed_list.append((complex_model_id, str(e)))

        return (complex_id, success_count, failed_list)

    except Exception as e:
        import traceback
        return (f"unknown_{row.name}", 0, [(f"unknown_{row.name}", f"Exception: {str(e)}\n{traceback.format_exc()}")])


class RNAPocketDataset(Dataset):
    """
    PyTorch Geometric Dataset for RNA binding pockets.

    This dataset loads pre-processed molecular graphs and their corresponding
    ligand embeddings for training.
    """

    def __init__(self, root, complex_ids, ligand_embeddings_path, transform=None, pre_transform=None):
        """
        Args:
            root: Root directory where graph files are stored
            complex_ids: List of complex IDs to include in dataset
            ligand_embeddings_path: Path to HDF5 file containing ligand embeddings
            transform: Optional transform to apply to data
            pre_transform: Optional pre-transform to apply to data
        """
        self.complex_ids = complex_ids
        self.ligand_embeddings_path = ligand_embeddings_path

        # Load ligand embeddings
        self.ligand_embeddings = {}
        with h5py.File(ligand_embeddings_path, 'r') as f:
            for complex_id in complex_ids:
                if complex_id in f:
                    self.ligand_embeddings[complex_id] = torch.tensor(
                        f[complex_id][:],
                        dtype=torch.float
                    )

        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """Required by PyTorch Geometric."""
        return []

    @property
    def processed_file_names(self):
        """List of processed graph files."""
        return [f"{complex_id}.pt" for complex_id in self.complex_ids]

    def download(self):
        """Not implemented - data should be pre-downloaded."""
        pass

    def process(self):
        """Not implemented - graphs should be pre-processed."""
        pass

    def len(self):
        """Return dataset size."""
        return len(self.complex_ids)

    def get(self, idx):
        """
        Load a single data point.

        Args:
            idx: Index of the data point

        Returns:
            Tuple of (graph_data, ligand_embedding)
        """
        complex_id = self.complex_ids[idx]

        # Load graph
        graph_path = Path(self.processed_dir) / f"{complex_id}.pt"
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")

        data = torch.load(graph_path)

        # Get ligand embedding
        if complex_id not in self.ligand_embeddings:
            raise KeyError(f"Ligand embedding not found for {complex_id}")

        ligand_embedding = self.ligand_embeddings[complex_id]

        # Store ligand embedding in data object
        data.y = ligand_embedding

        return data


def build_and_save_graphs(hariboss_csv, pocket_dir, amber_dir, output_dir, distance_cutoff=4.0, num_workers=None):
    """
    Build molecular graphs for RNA structures and save them using multiprocessing.

    Note: This function builds graphs from RNA-only PDB and topology files,
          NOT the full pocket (which includes ligands).

    Args:
        hariboss_csv: Path to HARIBOSS CSV file
        pocket_dir: Directory containing pocket PDB files (not used, kept for compatibility)
        amber_dir: Directory containing AMBER topology and RNA PDB files
        output_dir: Directory to save graph files
        distance_cutoff: Distance cutoff for edge construction
        num_workers: Number of parallel workers (default: CPU count)
    """
    # Read HARIBOSS CSV
    print(f"Reading HARIBOSS CSV from {hariboss_csv}...")
    hariboss_df = pd.read_csv(hariboss_csv)

    # Find PDB ID and ligand columns
    pdb_id_column = None
    for col in ['id', 'pdb_id', 'PDB_ID', 'pdbid', 'PDBID', 'PDB']:
        if col in hariboss_df.columns:
            pdb_id_column = col
            break

    if pdb_id_column is None:
        print("Error: Could not find PDB ID column")
        return

    ligand_column = None
    for col in ['sm_ligand_ids', 'ligand', 'Ligand', 'ligand_resname', 'LIGAND', 'ligand_name']:
        if col in hariboss_df.columns:
            ligand_column = col
            break

    if ligand_column is None:
        hariboss_df['ligand_resname'] = 'LIG'
        ligand_column = 'ligand_resname'

    # Setup paths
    pocket_dir = Path(pocket_dir)
    amber_dir = Path(amber_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()

    print(f"\n{'='*60}")
    print(f"Building RNA-only graphs for {len(hariboss_df)} complexes")
    print(f"Note: Graphs will include RNA atoms only, NOT ligands")
    print(f"Note: Processing all models for each complex")
    print(f"Using {num_workers} parallel workers")
    print(f"{'='*60}\n")

    # Create partial function with fixed parameters
    process_func = partial(
        process_single_complex,
        pdb_id_column=pdb_id_column,
        ligand_column=ligand_column,
        amber_dir=amber_dir,
        output_dir=output_dir,
        distance_cutoff=distance_cutoff
    )

    # Process with multiprocessing
    total_success = 0
    all_failed = []

    with Pool(processes=num_workers) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(process_func, [row for _, row in hariboss_df.iterrows()]),
            total=len(hariboss_df),
            desc="Processing complexes"
        ))

    # Aggregate results
    for complex_id, success_count, failed_list in results:
        total_success += success_count
        all_failed.extend(failed_list)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Graph construction complete!")
    print(f"Successfully built: {total_success} graphs")
    print(f"Failed: {len(all_failed)}")

    if all_failed:
        failed_df = pd.DataFrame(all_failed, columns=['complex_id', 'reason'])
        failed_path = output_dir.parent / "failed_graph_construction.csv"
        failed_df.to_csv(failed_path, index=False)
        print(f"Failed graphs saved to {failed_path}")

    # ========================================
    # Apply global feature normalization
    # ========================================
    if total_success > 0:
        print(f"\n{'='*60}")
        print(f"Computing global feature normalization parameters...")
        print(f"{'='*60}\n")

        # Collect all node features from saved graphs
        all_node_features = []
        graph_files = sorted(output_dir.glob("*.pt"))

        print(f"Loading {len(graph_files)} graphs to collect features...")
        for graph_path in tqdm(graph_files, desc="Collecting features"):
            try:
                data = torch.load(graph_path)
                if hasattr(data, 'x') and data.x is not None:
                    all_node_features.append(data.x.numpy())
            except Exception as e:
                print(f"Warning: Failed to load {graph_path}: {e}")

        if len(all_node_features) > 0:
            # Concatenate all features
            all_features = np.vstack(all_node_features)
            print(f"Collected features from {len(all_node_features)} graphs")
            print(f"Total atoms: {all_features.shape[0]}, Feature dim: {all_features.shape[1]}")

            # Feature vector: [atom_type_idx, charge, residue_idx, atomic_num]
            # Only normalize continuous features (indices 1 and 3)
            # DO NOT normalize discrete indices (0 and 2)

            continuous_indices = [1, 3]  # charge and atomic_num

            # Compute statistics only for continuous features
            feature_mean = np.zeros(all_features.shape[1])
            feature_std = np.ones(all_features.shape[1])

            feature_mean[continuous_indices] = np.mean(all_features[:, continuous_indices], axis=0)
            feature_std[continuous_indices] = np.std(all_features[:, continuous_indices], axis=0)

            # Add small epsilon to avoid division by zero
            feature_std[continuous_indices] = np.where(
                feature_std[continuous_indices] < 1e-8,
                1.0,
                feature_std[continuous_indices]
            )

            print(f"\nNormalization parameters (only for continuous features):")
            print(f"  Charge (idx 1): mean={feature_mean[1]:.4f}, std={feature_std[1]:.4f}")
            print(f"  Atomic num (idx 3): mean={feature_mean[3]:.4f}, std={feature_std[3]:.4f}")
            print(f"  Note: Discrete indices (0=atom_type, 2=residue) are NOT normalized")

            # Save normalization parameters
            norm_params_path = output_dir.parent / "node_feature_norm_params.npz"
            np.savez(norm_params_path,
                     mean=feature_mean,
                     std=feature_std,
                     continuous_indices=continuous_indices)
            print(f"Saved normalization parameters to {norm_params_path}")

            # Apply normalization to all graphs and resave
            print(f"\nApplying normalization to all graphs...")
            for graph_path in tqdm(graph_files, desc="Normalizing graphs"):
                try:
                    data = torch.load(graph_path)
                    if hasattr(data, 'x') and data.x is not None:
                        # Apply normalization only to continuous features
                        x_normalized = data.x.numpy().copy()
                        x_normalized[:, continuous_indices] = (
                            x_normalized[:, continuous_indices] - feature_mean[continuous_indices]
                        ) / feature_std[continuous_indices]
                        data.x = torch.from_numpy(x_normalized.astype(np.float32))
                        # Save normalized graph
                        torch.save(data, graph_path)
                except Exception as e:
                    print(f"Warning: Failed to normalize {graph_path}: {e}")

            print(f"Feature normalization complete!")
        else:
            print(f"Warning: No features collected, skipping normalization")


def main():
    """Main graph construction pipeline."""
    parser = argparse.ArgumentParser(description="Build molecular graphs for RNA pockets")
    parser.add_argument("--hariboss_csv", type=str, default="hariboss/Complexes.csv",
                        help="Path to HARIBOSS complexes CSV file")
    parser.add_argument("--pocket_dir", type=str, default="data/processed/pockets",
                        help="Directory containing pocket PDB files")
    parser.add_argument("--amber_dir", type=str, default="data/processed/amber",
                        help="Directory containing AMBER topology files")
    parser.add_argument("--output_dir", type=str, default="data/processed/graphs",
                        help="Output directory for graph files")
    parser.add_argument("--distance_cutoff", type=float, default=4.0,
                        help="Distance cutoff for edge construction (Angstroms)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")

    args = parser.parse_args()

    build_and_save_graphs(
        args.hariboss_csv,
        args.pocket_dir,
        args.amber_dir,
        args.output_dir,
        args.distance_cutoff,
        args.num_workers
    )


if __name__ == "__main__":
    main()
