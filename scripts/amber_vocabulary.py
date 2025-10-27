#!/usr/bin/env python3
"""
AMBER Force Field Vocabulary Utilities

This module provides fixed vocabularies for AMBER atom types and RNA residues
to ensure consistent feature dimensions across different datasets.
"""

from pathlib import Path
from typing import Dict, List
import numpy as np
import json


def load_amber_atom_types(vocab_file: str = None) -> List[str]:
    """
    Load AMBER atom type vocabulary from file.

    Args:
        vocab_file: Path to vocabulary file. If None, uses default location.

    Returns:
        List of AMBER atom types in canonical order
    """
    if vocab_file is None:
        # Default location
        vocab_file = Path(__file__).parent.parent / "data" / "amber_rna_atom_types.txt"

    atom_types = []

    with open(vocab_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Extract atom type (first column)
            parts = line.split()
            if len(parts) >= 1:
                atom_type = parts[0]
                atom_types.append(atom_type)

    return atom_types


def get_rna_residue_types() -> List[str]:
    """
    Get standard RNA residue types.

    Returns:
        List of RNA residue names in canonical order
    """
    # Standard nucleotides
    standard_residues = [
        'A', 'G', 'C', 'U',          # Standard bases
        'A5', 'G5', 'C5', 'U5',      # 5' terminal
        'A3', 'G3', 'C3', 'U3',      # 3' terminal
        'RA', 'RG', 'RC', 'RU',      # Alternative naming
        'DA', 'DG', 'DC', 'DT',      # DNA (if present)
        'ADE', 'GUA', 'CYT', 'URA',  # Full names
    ]

    # Common modified nucleotides
    modified_residues = [
        'PSU', 'I', 'M2G', 'M7G', 'OMC', 'OMG',  # Common modifications
        '5MU', '5MC', '1MA', '2MG', '6MA',        # Methylations
    ]

    # Ions and cofactors
    ions = [
        'MG', 'K', 'NA', 'CA', 'ZN', 'MN', 'CL'  # Common ions
    ]

    return standard_residues + modified_residues + ions


class AMBERFeatureEncoder:
    """
    Encoder for AMBER-based node features with fixed vocabularies.
    """

    def __init__(self, atom_type_vocab: List[str] = None, residue_vocab: List[str] = None):
        """
        Args:
            atom_type_vocab: List of AMBER atom types. If None, loads from file.
            residue_vocab: List of residue names. If None, uses default RNA residues.
        """
        # Load or use provided vocabularies
        if atom_type_vocab is None:
            self.atom_type_vocab = load_amber_atom_types()
        else:
            self.atom_type_vocab = atom_type_vocab

        if residue_vocab is None:
            self.residue_vocab = get_rna_residue_types()
        else:
            self.residue_vocab = residue_vocab

        # Create lookup dictionaries
        self.atom_type_to_idx = {atype: idx for idx, atype in enumerate(self.atom_type_vocab)}
        self.residue_to_idx = {resname: idx for idx, resname in enumerate(self.residue_vocab)}

        # Add special tokens
        self.atom_type_to_idx['<UNK>'] = len(self.atom_type_vocab)  # Unknown atom type
        self.residue_to_idx['<UNK>'] = len(self.residue_vocab)      # Unknown residue

        # Vocabulary sizes (including <UNK>)
        self.num_atom_types = len(self.atom_type_vocab) + 1
        self.num_residues = len(self.residue_vocab) + 1

    def encode_atom_type(self, atom_type: str) -> int:
        """
        Encode AMBER atom type as integer index (0-indexed).

        Args:
            atom_type: AMBER atom type string

        Returns:
            Integer index (0 to num_atom_types-1)
        """
        idx = self.atom_type_to_idx.get(atom_type, self.atom_type_to_idx['<UNK>'])
        return idx  # 0-indexed

    def encode_residue(self, residue_name: str) -> int:
        """
        Encode residue name as integer index (0-indexed).

        Args:
            residue_name: Residue name string

        Returns:
            Integer index (0 to num_residues-1)
        """
        idx = self.residue_to_idx.get(residue_name, self.residue_to_idx['<UNK>'])
        return idx  # 0-indexed

    def encode_atom_features(self, atom_type: str, charge: float, residue_name: str,
                            atomic_number: int) -> np.ndarray:
        """
        Encode complete atom features as integer indices + scalars.

        Args:
            atom_type: AMBER atom type
            charge: Partial charge
            residue_name: Residue name
            atomic_number: Atomic number

        Returns:
            Feature vector: [atom_type_idx, charge, residue_idx, atomic_number] (4 dims)
        """
        atom_type_idx = self.encode_atom_type(atom_type)
        residue_idx = self.encode_residue(residue_name)

        # Return as 4-dimensional feature vector
        features = np.array([
            float(atom_type_idx),    # 0-69 (0-68 normal, 69 UNK)
            float(charge),            # scalar
            float(residue_idx),       # 0-42 (0-41 normal, 42 UNK)
            float(atomic_number)      # scalar
        ], dtype=np.float32)

        return features

    @property
    def feature_dim(self) -> int:
        """Total feature dimension."""
        return 4  # [atom_type_idx, charge, residue_idx, atomic_num]

    def save_vocabularies(self, output_dir: str):
        """
        Save atom type and residue vocabularies to JSON files.

        Args:
            output_dir: Directory to save vocabulary files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save atom type vocabulary
        atom_type_vocab = {
            'vocab': self.atom_type_vocab,
            'vocab_to_idx': {k: v for k, v in self.atom_type_to_idx.items() if k != '<UNK>'},
            'idx_to_vocab': {v: k for k, v in self.atom_type_to_idx.items() if k != '<UNK>'},
            'num_types': len(self.atom_type_vocab),
            'unk_idx': self.atom_type_to_idx['<UNK>']
        }

        with open(output_path / 'atom_type_vocab.json', 'w') as f:
            json.dump(atom_type_vocab, f, indent=2)

        # Save residue vocabulary
        residue_vocab = {
            'vocab': self.residue_vocab,
            'vocab_to_idx': {k: v for k, v in self.residue_to_idx.items() if k != '<UNK>'},
            'idx_to_vocab': {v: k for k, v in self.residue_to_idx.items() if k != '<UNK>'},
            'num_types': len(self.residue_vocab),
            'unk_idx': self.residue_to_idx['<UNK>']
        }

        with open(output_path / 'residue_vocab.json', 'w') as f:
            json.dump(residue_vocab, f, indent=2)

        print(f"Saved vocabularies to {output_path}")
        print(f"  - atom_type_vocab.json: {len(self.atom_type_vocab)} types")
        print(f"  - residue_vocab.json: {len(self.residue_vocab)} types")

    def __repr__(self):
        return (f"AMBERFeatureEncoder("
                f"num_atom_types={self.num_atom_types}, "
                f"num_residues={self.num_residues}, "
                f"feature_dim={self.feature_dim})")


# Global encoder instance (singleton pattern)
_global_encoder = None


def get_global_encoder() -> AMBERFeatureEncoder:
    """
    Get the global feature encoder instance (singleton).

    Returns:
        Shared AMBERFeatureEncoder instance
    """
    global _global_encoder
    if _global_encoder is None:
        _global_encoder = AMBERFeatureEncoder()
    return _global_encoder


if __name__ == "__main__":
    # Test the encoder
    print("=" * 80)
    print("AMBER Feature Encoder Test")
    print("=" * 80)

    encoder = get_global_encoder()
    print(f"\n{encoder}")
    print(f"\nAtom types loaded: {len(encoder.atom_type_vocab)}")
    print(f"Sample atom types: {encoder.atom_type_vocab[:10]}")
    print(f"\nResidue types loaded: {len(encoder.residue_vocab)}")
    print(f"Sample residues: {encoder.residue_vocab[:10]}")

    # Test encoding
    print("\n" + "=" * 80)
    print("Encoding Test")
    print("=" * 80)

    test_features = encoder.encode_atom_features(
        atom_type="C4",
        charge=-0.123,
        residue_name="G",
        atomic_number=6
    )

    print(f"\nTest atom: C4, charge=-0.123, residue=G, atomic_num=6")
    print(f"Encoded features shape: {test_features.shape}")
    print(f"Feature vector (first 20): {test_features[:20]}")

    # Test unknown handling
    test_unknown = encoder.encode_atom_features(
        atom_type="UNKNOWN_TYPE",
        charge=0.0,
        residue_name="XYZ",
        atomic_number=99
    )
    print(f"\nUnknown test:")
    print(f"Encoded features shape: {test_unknown.shape}")
    print(f"<UNK> positions should have value 1.0")

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
