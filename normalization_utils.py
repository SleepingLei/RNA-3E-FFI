#!/usr/bin/env python3
"""
Normalization Utilities

This module provides utilities for loading and applying normalization parameters
saved during training to ensure consistent preprocessing during inference/testing.
"""
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Union


def load_node_feature_norm_params(params_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load node feature normalization parameters saved during training.

    Args:
        params_path: Path to the .npz file containing normalization parameters

    Returns:
        Tuple of (mean, std) as numpy arrays

    Raises:
        FileNotFoundError: If the params file doesn't exist
        KeyError: If required keys are not in the file
    """
    params_path = Path(params_path)
    if not params_path.exists():
        raise FileNotFoundError(f"Normalization parameters not found at {params_path}")

    params = np.load(params_path)

    if 'mean' not in params or 'std' not in params:
        raise KeyError("Normalization parameters file must contain 'mean' and 'std' keys")

    mean = params['mean']
    std = params['std']

    return mean, std


def load_ligand_embedding_norm_params(params_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ligand embedding normalization parameters saved during training.

    Args:
        params_path: Path to the .npz file containing normalization parameters

    Returns:
        Tuple of (mean, std) as numpy arrays

    Raises:
        FileNotFoundError: If the params file doesn't exist
        KeyError: If required keys are not in the file
    """
    return load_node_feature_norm_params(params_path)


def normalize_node_features(features: Union[np.ndarray, torch.Tensor],
                            mean: np.ndarray,
                            std: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply z-score normalization to node features using pre-computed parameters.

    Args:
        features: Node features to normalize (N x D array/tensor)
        mean: Mean values for each feature dimension (D,)
        std: Standard deviation for each feature dimension (D,)

    Returns:
        Normalized features in the same format as input
    """
    is_torch = isinstance(features, torch.Tensor)

    if is_torch:
        features_np = features.numpy()
    else:
        features_np = features

    # Apply normalization
    normalized = (features_np - mean) / std

    if is_torch:
        return torch.from_numpy(normalized.astype(np.float32))
    else:
        return normalized.astype(np.float32)


def normalize_ligand_embedding(embedding: Union[np.ndarray, torch.Tensor],
                               mean: np.ndarray,
                               std: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply z-score normalization to ligand embedding using pre-computed parameters.

    Args:
        embedding: Ligand embedding to normalize (D,) or (1, D)
        mean: Mean values for each embedding dimension (D,)
        std: Standard deviation for each embedding dimension (D,)

    Returns:
        Normalized embedding in the same format as input
    """
    return normalize_node_features(embedding, mean, std)


def denormalize_ligand_embedding(normalized_embedding: Union[np.ndarray, torch.Tensor],
                                 mean: np.ndarray,
                                 std: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
    """
    Reverse z-score normalization to get original ligand embedding values.

    This is useful when you need to interpret or compare embeddings in their original scale.

    Args:
        normalized_embedding: Normalized embedding (D,) or (1, D)
        mean: Mean values used for normalization (D,)
        std: Standard deviation used for normalization (D,)

    Returns:
        Original (denormalized) embedding in the same format as input
    """
    is_torch = isinstance(normalized_embedding, torch.Tensor)

    if is_torch:
        embedding_np = normalized_embedding.numpy()
    else:
        embedding_np = normalized_embedding

    # Reverse normalization: x_original = x_normalized * std + mean
    denormalized = embedding_np * std + mean

    if is_torch:
        return torch.from_numpy(denormalized.astype(np.float32))
    else:
        return denormalized.astype(np.float32)


class NormalizationContext:
    """
    Context manager for loading and using normalization parameters.

    Example usage:
        >>> with NormalizationContext('data/processed') as norm:
        ...     normalized_features = norm.normalize_features(node_features)
        ...     normalized_embedding = norm.normalize_embedding(ligand_embedding)
    """

    def __init__(self, processed_dir: Union[str, Path]):
        """
        Initialize normalization context.

        Args:
            processed_dir: Directory containing normalization parameter files
        """
        self.processed_dir = Path(processed_dir)
        self.node_feature_mean = None
        self.node_feature_std = None
        self.ligand_embedding_mean = None
        self.ligand_embedding_std = None

    def __enter__(self):
        """Load normalization parameters."""
        # Load node feature normalization params
        node_params_path = self.processed_dir / "node_feature_norm_params.npz"
        if node_params_path.exists():
            self.node_feature_mean, self.node_feature_std = load_node_feature_norm_params(node_params_path)
            print(f"Loaded node feature normalization parameters from {node_params_path}")
        else:
            print(f"Warning: Node feature normalization parameters not found at {node_params_path}")

        # Load ligand embedding normalization params
        ligand_params_path = self.processed_dir / "ligand_embedding_norm_params.npz"
        if ligand_params_path.exists():
            self.ligand_embedding_mean, self.ligand_embedding_std = load_ligand_embedding_norm_params(ligand_params_path)
            print(f"Loaded ligand embedding normalization parameters from {ligand_params_path}")
        else:
            print(f"Warning: Ligand embedding normalization parameters not found at {ligand_params_path}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup (nothing to do)."""
        pass

    def normalize_features(self, features: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Normalize node features using loaded parameters."""
        if self.node_feature_mean is None or self.node_feature_std is None:
            raise ValueError("Node feature normalization parameters not loaded")
        return normalize_node_features(features, self.node_feature_mean, self.node_feature_std)

    def normalize_embedding(self, embedding: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Normalize ligand embedding using loaded parameters."""
        if self.ligand_embedding_mean is None or self.ligand_embedding_std is None:
            raise ValueError("Ligand embedding normalization parameters not loaded")
        return normalize_ligand_embedding(embedding, self.ligand_embedding_mean, self.ligand_embedding_std)

    def denormalize_embedding(self, normalized_embedding: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Denormalize ligand embedding using loaded parameters."""
        if self.ligand_embedding_mean is None or self.ligand_embedding_std is None:
            raise ValueError("Ligand embedding normalization parameters not loaded")
        return denormalize_ligand_embedding(normalized_embedding, self.ligand_embedding_mean, self.ligand_embedding_std)


if __name__ == "__main__":
    # Example usage
    print("Normalization Utilities")
    print("=" * 60)
    print("\nExample usage:")
    print("""
    # Load normalization parameters
    node_mean, node_std = load_node_feature_norm_params('data/processed/node_feature_norm_params.npz')
    ligand_mean, ligand_std = load_ligand_embedding_norm_params('data/processed/ligand_embedding_norm_params.npz')

    # Normalize new data
    normalized_features = normalize_node_features(node_features, node_mean, node_std)
    normalized_embedding = normalize_ligand_embedding(ligand_embedding, ligand_mean, ligand_std)

    # Or use context manager
    with NormalizationContext('data/processed') as norm:
        normalized_features = norm.normalize_features(node_features)
        normalized_embedding = norm.normalize_embedding(ligand_embedding)
    """)
