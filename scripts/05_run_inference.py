#!/usr/bin/env python3
"""
Inference Script for RNA Pocket Encoder V2/V3

This script uses the trained v2/v3 model to predict pocket embeddings and
find similar ligands from a pre-computed library.
"""
import os
import sys
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import h5py
from tqdm import tqdm
import warnings

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import V3 improvements
try:
    from models.e3_gnn_encoder_v3 import RNAPocketEncoderV3
    _has_v3_model = True
except ImportError:
    _has_v3_model = False
    warnings.warn("V3 model not available. Only V2 model will be supported.")

# V2 models (backward compatible)
try:
    from models.e3_gnn_encoder_v2_fixed import RNAPocketEncoderV2Fixed
    _has_v2_fixed = True
except ImportError:
    _has_v2_fixed = False

from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2


def load_model(checkpoint_path, device):
    """
    Load trained v2/v3 model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model and its configuration
    """
    print(f"Loading model from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config from checkpoint or config.json
    config = checkpoint.get('config', {})

    # If config is missing or incomplete, try loading from config.json
    checkpoint_dir = Path(checkpoint_path).parent
    config_json_path = checkpoint_dir / 'config.json'

    if config_json_path.exists():
        print(f"  Loading config from {config_json_path}...")
        import json
        with open(config_json_path, 'r') as f:
            file_config = json.load(f)
            # Merge configs, preferring file config for model architecture params
            config_keys = [
                'input_dim', 'feature_hidden_dim', 'hidden_irreps',
                'output_dim', 'num_layers', 'num_radial_basis',
                'use_multi_hop', 'use_nonbonded', 'use_gate',
                'pooling_type', 'use_layer_norm', 'dropout',
                # V3-specific
                'use_v3_model', 'use_enhanced_invariants', 'use_improved_layers',
                'norm_type', 'num_attention_heads',
                'initial_angle_weight', 'initial_dihedral_weight', 'initial_nonbonded_weight',
                'use_weight_constraints'
            ]
            for key in config_keys:
                if key in file_config:
                    config[key] = file_config[key]
    elif not config:
        raise ValueError("Checkpoint does not contain model configuration and no config.json found. "
                       "Please use a checkpoint saved with v2/v3 training script.")

    # Determine model version
    use_v3 = config.get('use_v3_model', False)
    use_weight_constraints = config.get('use_weight_constraints', False)

    print(f"  Model version: {'V3' if use_v3 else 'V2' + (' (Fixed)' if use_weight_constraints else '')}")
    print(f"  Training epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Validation loss: {checkpoint.get('val_loss', 'unknown')}")

    # Initialize model with config from checkpoint
    if use_v3 and _has_v3_model:
        print("  Using RNAPocketEncoderV3")

        # V3-specific pooling type override
        pooling_type = config.get('pooling_type', 'attention')
        num_attention_heads = config.get('num_attention_heads', 4)
        if num_attention_heads > 1:
            pooling_type = 'multihead_attention'

        model = RNAPocketEncoderV3(
            input_dim=config.get('input_dim', 3),
            feature_hidden_dim=config.get('feature_hidden_dim', 64),
            hidden_irreps=config.get('hidden_irreps', '32x0e + 16x1o + 8x2e'),
            output_dim=config.get('output_dim', 1536),
            num_layers=config.get('num_layers', 6),
            num_radial_basis=config.get('num_radial_basis', 8),
            use_multi_hop=config.get('use_multi_hop', True),
            use_nonbonded=config.get('use_nonbonded', True),
            use_gate=config.get('use_gate', True),
            use_layer_norm=config.get('use_layer_norm', True),
            pooling_type=pooling_type,
            dropout=config.get('dropout', 0.1),
            # V3-specific parameters
            use_enhanced_invariants=config.get('use_enhanced_invariants', False),
            num_attention_heads=num_attention_heads,
            initial_angle_weight=config.get('initial_angle_weight', 0.5),
            initial_dihedral_weight=config.get('initial_dihedral_weight', 0.5),
            initial_nonbonded_weight=config.get('initial_nonbonded_weight', 0.5),
            use_improved_layers=config.get('use_improved_layers', False),
            norm_type=config.get('norm_type', 'layer')
        )
    elif use_weight_constraints and _has_v2_fixed:
        print("  Using RNAPocketEncoderV2Fixed (with weight constraints)")
        model = RNAPocketEncoderV2Fixed(
            input_dim=config.get('input_dim', 3),
            feature_hidden_dim=config.get('feature_hidden_dim', 64),
            hidden_irreps=config.get('hidden_irreps', '32x0e + 16x1o + 8x2e'),
            output_dim=config.get('output_dim', 512),
            num_layers=config.get('num_layers', 4),
            num_radial_basis=config.get('num_radial_basis', 8),
            use_multi_hop=config.get('use_multi_hop', True),
            use_nonbonded=config.get('use_nonbonded', True),
            use_gate=config.get('use_gate', True),
            use_layer_norm=config.get('use_layer_norm', False),
            pooling_type=config.get('pooling_type', 'attention'),
            dropout=config.get('dropout', 0.0)
        )
    else:
        print("  Using RNAPocketEncoderV2 (standard)")
        model = RNAPocketEncoderV2(
            input_dim=config.get('input_dim', 3),
            feature_hidden_dim=config.get('feature_hidden_dim', 64),
            hidden_irreps=config.get('hidden_irreps', '32x0e + 16x1o + 8x2e'),
            output_dim=config.get('output_dim', 512),
            num_layers=config.get('num_layers', 4),
            num_radial_basis=config.get('num_radial_basis', 8),
            use_multi_hop=config.get('use_multi_hop', True),
            use_nonbonded=config.get('use_nonbonded', True),
            use_gate=config.get('use_gate', True),
            use_layer_norm=config.get('use_layer_norm', False),
            pooling_type=config.get('pooling_type', 'attention'),
            dropout=config.get('dropout', 0.0)
        )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, config


def predict_pocket_embedding(pocket_graph, model, device):
    """
    Predict embedding for a single RNA pocket.

    Args:
        pocket_graph: PyTorch Geometric Data object for the pocket
        model: Trained model
        device: Device to run inference on

    Returns:
        Pocket embedding as numpy array
    """
    model.eval()

    with torch.no_grad():
        # Move graph to device
        pocket_graph = pocket_graph.to(device)

        # Forward pass
        embedding = model(pocket_graph)

        # Convert to numpy
        embedding = embedding.cpu().numpy()

    return embedding


def load_ligand_library(library_path):
    """
    Load pre-computed ligand embeddings from HDF5 file.

    Args:
        library_path: Path to HDF5 file containing ligand embeddings

    Returns:
        Dictionary mapping ligand IDs to embeddings
    """
    print(f"Loading ligand library from {library_path}...")

    ligand_library = {}

    with h5py.File(library_path, 'r') as f:
        # Iterate through all ligand IDs
        for ligand_id in f.keys():
            embedding = f[ligand_id][:]
            ligand_library[ligand_id] = embedding

    print(f"Loaded {len(ligand_library)} ligand embeddings")
    return ligand_library


def find_similar_ligands(query_embedding, ligand_library, top_k=10, metric='cosine'):
    """
    Find top-k most similar ligands based on embedding distance.

    Args:
        query_embedding: Query pocket embedding
        ligand_library: Dictionary of ligand embeddings
        top_k: Number of top matches to return
        metric: Distance metric ('cosine' or 'euclidean')

    Returns:
        List of (ligand_id, distance) tuples sorted by distance
    """
    distances = []

    for ligand_id, ligand_embedding in ligand_library.items():
        if metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            similarity = np.dot(query_embedding, ligand_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(ligand_embedding)
            )
            distance = 1 - similarity
        elif metric == 'euclidean':
            distance = np.linalg.norm(query_embedding - ligand_embedding)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        distances.append((ligand_id, distance))

    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[1])

    return distances[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained RNA pocket encoder")

    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--query_graph", type=str, required=True,
                        help="Path to query pocket graph (.pt file)")

    # Optional arguments
    parser.add_argument("--ligand_library", type=str, default=None,
                        help="Path to ligand embedding library (HDF5 file)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results (JSON)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top matches to return")
    parser.add_argument("--metric", type=str, default="cosine",
                        choices=['cosine', 'euclidean'],
                        help="Distance metric for similarity search")

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Load query pocket graph
    print(f"\nLoading query pocket from {args.query_graph}...")
    query_graph = torch.load(args.query_graph, weights_only=False)
    print(f"Query graph: {query_graph.num_nodes} nodes, {query_graph.num_edges} edges")

    # Predict query embedding
    print("\nPredicting pocket embedding...")
    query_embedding = predict_pocket_embedding(query_graph, model, device)
    print(f"Predicted embedding shape: {query_embedding.shape}")

    # If ligand library provided, find similar ligands
    if args.ligand_library:
        # Load ligand library
        ligand_library = load_ligand_library(args.ligand_library)

        # Find similar ligands
        print(f"\nFinding top-{args.top_k} similar ligands using {args.metric} distance...")
        similar_ligands = find_similar_ligands(
            query_embedding,
            ligand_library,
            top_k=args.top_k,
            metric=args.metric
        )

        # Print results
        print("\nTop matches:")
        print("-" * 60)
        for rank, (ligand_id, distance) in enumerate(similar_ligands, 1):
            print(f"{rank:2d}. {ligand_id:30s} Distance: {distance:.6f}")

        # Save results if output path provided
        if args.output:
            results = {
                'query': str(args.query_graph),
                'metric': args.metric,
                'top_k': args.top_k,
                'query_embedding': query_embedding.tolist(),
                'matches': [
                    {
                        'rank': rank,
                        'ligand_id': ligand_id,
                        'distance': float(distance)
                    }
                    for rank, (ligand_id, distance) in enumerate(similar_ligands, 1)
                ]
            }

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\nResults saved to {output_path}")

    else:
        # Just save the embedding
        if args.output:
            results = {
                'query': str(args.query_graph),
                'embedding': query_embedding.tolist()
            }

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\nEmbedding saved to {output_path}")


def batch_inference(checkpoint_path, graph_dir, output_path, device=None):
    """
    Run inference on multiple pocket graphs and save embeddings.

    Args:
        checkpoint_path: Path to model checkpoint
        graph_dir: Directory containing pocket graph files
        output_path: Path to save output embeddings (HDF5 or NPZ)
        device: Device to use (defaults to cuda if available)

    Example:
        batch_inference(
            checkpoint_path="models/checkpoints/best_model.pt",
            graph_dir="data/processed/",
            output_path="data/pocket_embeddings.h5"
        )
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Running batch inference on {graph_dir}")
    print(f"Using device: {device}")

    # Load model
    model, config = load_model(checkpoint_path, device)

    # Find all graph files
    graph_files = list(Path(graph_dir).glob("*.pt"))
    print(f"Found {len(graph_files)} pocket graphs")

    # Predict embeddings
    embeddings = {}

    for graph_file in tqdm(graph_files, desc="Processing"):
        try:
            # Extract complex ID from filename
            complex_id = graph_file.stem

            # Load graph
            graph = torch.load(graph_file, weights_only=False)

            # Predict embedding
            embedding = predict_pocket_embedding(graph, model, device)

            embeddings[complex_id] = embedding

        except Exception as e:
            print(f"Error processing {graph_file}: {e}")
            continue

    # Save embeddings
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == '.h5':
        # Save as HDF5
        with h5py.File(output_path, 'w') as f:
            for complex_id, embedding in embeddings.items():
                f.create_dataset(complex_id, data=embedding)
        print(f"Saved {len(embeddings)} embeddings to {output_path}")

    elif output_path.suffix == '.npz':
        # Save as NPZ
        np.savez(output_path, **embeddings)
        print(f"Saved {len(embeddings)} embeddings to {output_path}")

    else:
        raise ValueError(f"Unknown output format: {output_path.suffix}. Use .h5 or .npz")

    return embeddings


if __name__ == "__main__":
    main()
