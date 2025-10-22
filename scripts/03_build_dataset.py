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


def build_graph_from_files(rna_pdb_path, prmtop_path, distance_cutoff=4.0):
    """
    Build a molecular graph from RNA PDB and AMBER topology files.

    This function creates a consistent atom ordering using the PDB file as reference,
    then extracts features from both RDKit and ParmEd.

    Args:
        rna_pdb_path: Path to RNA PDB file
        prmtop_path: Path to AMBER prmtop file
        distance_cutoff: Distance cutoff for edge construction (Angstroms)

    Returns:
        torch_geometric.data.Data object with node features, positions, and edges
    """
    try:
        # Load RNA with RDKit
        mol = Chem.MolFromPDBFile(str(rna_pdb_path), removeHs=False, sanitize=False)
        if mol is None:
            print(f"Error: RDKit failed to load {rna_pdb_path}")
            return None

        # Try to sanitize
        try:
            Chem.SanitizeMol(mol)
        except:
            print(f"Warning: Sanitization failed for {rna_pdb_path}")

        # Load AMBER topology with ParmEd
        amber_parm = pmd.load_file(str(prmtop_path))

        # Get number of atoms
        n_atoms = mol.GetNumAtoms()
        n_amber_atoms = len(amber_parm.atoms)

        if n_atoms != n_amber_atoms:
            print(f"Warning: Atom count mismatch - RDKit: {n_atoms}, AMBER: {n_amber_atoms}")
            # Use minimum to avoid index errors
            n_atoms = min(n_atoms, n_amber_atoms)

        # Create atom mapping: (resname, resid, atom_name) -> index
        # Use RDKit PDB info as reference
        atom_mapping = {}
        for i, atom in enumerate(mol.GetAtoms()):
            pdb_info = atom.GetPDBResidueInfo()
            if pdb_info:
                key = (
                    pdb_info.GetResidueName().strip(),
                    pdb_info.GetResidueNumber(),
                    pdb_info.GetName().strip()
                )
                atom_mapping[key] = i

        # Extract features from RDKit
        rdkit_features = []
        for i in range(n_atoms):
            atom = mol.GetAtomWithIdx(i)

            # Atomic number (one-hot encoding for common elements)
            atomic_num = atom.GetAtomicNum()

            # Hybridization
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

        # Extract features from ParmEd (partial charges and atom types)
        amber_features = []
        for i in range(n_atoms):
            atom = amber_parm.atoms[i]

            # Partial charge
            charge = float(atom.charge)

            # AMBER atom type (convert to hash for now, or could use embedding)
            atom_type_hash = hash(atom.type) % 1000  # Simple hash

            amber_features.append([charge, atom_type_hash])

        amber_features = np.array(amber_features, dtype=np.float32)

        # Concatenate features
        node_features = np.concatenate([rdkit_features, amber_features], axis=1)
        x = torch.tensor(node_features, dtype=torch.float)

        # Extract 3D coordinates from RDKit conformer
        conf = mol.GetConformer()
        positions = []
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])

        pos = torch.tensor(positions, dtype=torch.float)

        # Build edges based on distance cutoff
        edge_index = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(positions[i] - np.array(positions[j]))
                if dist <= distance_cutoff:
                    # Add both directions (undirected graph)
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        if len(edge_index) == 0:
            print(f"Warning: No edges found with cutoff {distance_cutoff}")
            # Add self-loops as fallback
            edge_index = [[i, i] for i in range(n_atoms)]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Create PyTorch Geometric Data object
        data = Data(x=x, pos=pos, edge_index=edge_index)

        return data

    except Exception as e:
        print(f"Error building graph from {rna_pdb_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


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


def build_and_save_graphs(hariboss_csv, pocket_dir, amber_dir, output_dir, distance_cutoff=4.0):
    """
    Build molecular graphs for RNA structures and save them.

    Note: This function builds graphs from RNA-only PDB and topology files,
          NOT the full pocket (which includes ligands).

    Args:
        hariboss_csv: Path to HARIBOSS CSV file
        pocket_dir: Directory containing pocket PDB files (not used, kept for compatibility)
        amber_dir: Directory containing AMBER topology and RNA PDB files
        output_dir: Directory to save graph files
        distance_cutoff: Distance cutoff for edge construction
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

    # Build graphs
    success_count = 0
    failed_graphs = []

    print(f"\n{'='*60}")
    print(f"Building RNA-only graphs for {len(hariboss_df)} complexes")
    print(f"Note: Graphs will include RNA atoms only, NOT ligands")
    print(f"{'='*60}\n")
    for idx, row in tqdm(hariboss_df.iterrows(), total=len(hariboss_df)):
        pdb_id = str(row[pdb_id_column]).lower()

        # Parse ligand name from sm_ligand_ids if needed
        ligand_str = str(row[ligand_column])
        if ligand_column == 'sm_ligand_ids':
            # Parse format like "['ARG_.:B/1:N']" or "ARG_.:B/1:N"
            import ast
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

        # Check if graph already exists
        graph_path = output_dir / f"{complex_id}.pt"
        if graph_path.exists():
            success_count += 1
            continue

        # Define file paths - use RNA-only files
        rna_pdb_path = amber_dir / f"{complex_id}_rna.pdb"
        rna_prmtop_path = amber_dir / f"{complex_id}_rna.prmtop"

        # Check if files exist
        if not rna_pdb_path.exists():
            failed_graphs.append((complex_id, "rna_pdb_not_found"))
            continue

        if not rna_prmtop_path.exists():
            failed_graphs.append((complex_id, "rna_prmtop_not_found"))
            continue

        # Build graph
        try:
            data = build_graph_from_files(rna_pdb_path, rna_prmtop_path, distance_cutoff)

            if data is not None:
                # Save graph
                torch.save(data, graph_path)
                success_count += 1
            else:
                failed_graphs.append((complex_id, "graph_construction_failed"))

        except Exception as e:
            print(f"Error processing {complex_id}: {e}")
            failed_graphs.append((complex_id, str(e)))

    # Print summary
    print(f"\n{'='*60}")
    print(f"Graph construction complete!")
    print(f"Successfully built: {success_count}/{len(hariboss_df)}")
    print(f"Failed: {len(failed_graphs)}")

    if failed_graphs:
        failed_df = pd.DataFrame(failed_graphs, columns=['complex_id', 'reason'])
        failed_path = output_dir.parent / "failed_graph_construction.csv"
        failed_df.to_csv(failed_path, index=False)
        print(f"Failed graphs saved to {failed_path}")


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

    args = parser.parse_args()

    build_and_save_graphs(
        args.hariboss_csv,
        args.pocket_dir,
        args.amber_dir,
        args.output_dir,
        args.distance_cutoff
    )


if __name__ == "__main__":
    main()
