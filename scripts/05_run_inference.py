#!/usr/bin/env python3
"""
Inference Script for RNA Pocket Encoder

This script uses the trained model to predict pocket embeddings and
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

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.e3_gnn_encoder import RNAPocketEncoder


def load_model(checkpoint_path, input_dim, hidden_irreps, output_dim, num_layers, num_radial_basis, device):
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        input_dim: Input feature dimension
        hidden_irreps: Hidden layer irreps
        output_dim: Output embedding dimension
        num_layers: Number of message passing layers
        num_radial_basis: Number of radial basis functions
        device: Device to load model on

    Returns:
        Loaded model
    """
    print(f"Loading model from {checkpoint_path}...")

    # Initialize model
    model = RNAPocketEncoder(
        input_dim=input_dim,
        hidden_irreps=hidden_irreps,
        output_dim=output_dim,
        num_layers=num_layers,
        num_radial_basis=num_radial_basis
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    return model


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
        embedding_np = embedding.cpu().numpy()

        # Handle batch dimension
        if len(embedding_np.shape) > 1 and embedding_np.shape[0] == 1:
            embedding_np = embedding_np[0]

    return embedding_np


def calculate_distance(embedding1, embedding2, metric='euclidean'):
    """
    Calculate distance between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        metric: Distance metric ('euclidean' or 'cosine')

    Returns:
        Distance value
    """
    if metric == 'euclidean':
        return np.linalg.norm(embedding1 - embedding2)
    elif metric == 'cosine':
        # Cosine distance = 1 - cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        cosine_sim = dot_product / (norm1 * norm2 + 1e-8)
        return 1 - cosine_sim
    else:
        raise ValueError(f"Unknown metric: {metric}")


def find_similar_ligands(query_embedding, ligand_library, top_k=10, metric='euclidean'):
    """
    Find the top-k most similar ligands from a library.

    Args:
        query_embedding: Query pocket embedding
        ligand_library: Dictionary mapping ligand IDs to embeddings
        top_k: Number of top matches to return
        metric: Distance metric to use

    Returns:
        List of tuples (ligand_id, distance) sorted by distance
    """
    distances = []

    for ligand_id, ligand_embedding in ligand_library.items():
        dist = calculate_distance(query_embedding, ligand_embedding, metric)
        distances.append((ligand_id, dist))

    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[1])

    return distances[:top_k]


def load_ligand_library(embeddings_path):
    """
    Load pre-computed ligand embeddings.

    Args:
        embeddings_path: Path to HDF5 file containing ligand embeddings

    Returns:
        Dictionary mapping complex IDs to embedding vectors
    """
    print(f"Loading ligand library from {embeddings_path}...")

    ligand_library = {}
    with h5py.File(embeddings_path, 'r') as f:
        for complex_id in tqdm(f.keys(), desc="Loading embeddings"):
            ligand_library[complex_id] = np.array(f[complex_id][:])

    print(f"Loaded {len(ligand_library)} ligand embeddings")
    return ligand_library


def main():
    """Main inference pipeline."""
    parser = argparse.ArgumentParser(description="Run inference with trained RNA Pocket Encoder")

    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config JSON (optional, will use defaults if not provided)")

    # Data arguments
    parser.add_argument("--query_graph", type=str, required=True,
                        help="Path to query pocket graph (.pt file)")
    parser.add_argument("--ligand_library", type=str, required=True,
                        help="Path to ligand embeddings HDF5 file")

    # Inference arguments
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top matches to return")
    parser.add_argument("--metric", type=str, default='euclidean',
                        choices=['euclidean', 'cosine'],
                        help="Distance metric to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (JSON)")

    # Model architecture (defaults, can be overridden by config)
    parser.add_argument("--input_dim", type=int, default=11,
                        help="Input feature dimension")
    parser.add_argument("--hidden_irreps", type=str, default="32x0e + 16x1o + 8x2e",
                        help="Hidden layer irreps")
    parser.add_argument("--output_dim", type=int, default=512,
                        help="Output embedding dimension")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of message passing layers")
    parser.add_argument("--num_radial_basis", type=int, default=8,
                        help="Number of radial basis functions")

    args = parser.parse_args()

    # Load config if provided
    if args.config and Path(args.config).exists():
        print(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(
        args.checkpoint,
        args.input_dim,
        args.hidden_irreps,
        args.output_dim,
        args.num_layers,
        args.num_radial_basis,
        device
    )

    # Load query pocket graph
    print(f"\nLoading query pocket from {args.query_graph}...")
    query_graph = torch.load(args.query_graph)
    print(f"Query graph: {query_graph.num_nodes} nodes, {query_graph.num_edges} edges")

    # Predict query embedding
    print("\nPredicting pocket embedding...")
    query_embedding = predict_pocket_embedding(query_graph, model, device)
    print(f"Predicted embedding shape: {query_embedding.shape}")

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


def batch_inference_example():
    """
    Example function demonstrating batch inference on multiple pockets.
    """
    print("This is an example function for batch inference.")
    print("To use it, modify the code below with your specific paths and requirements.")

    # Example usage:
    # checkpoint_path = "models/checkpoints/best_model.pt"
    # graph_dir = "data/processed/graphs"
    # ligand_library_path = "data/processed/ligand_embeddings.h5"
    # output_dir = "results/predictions"

    # # Load model
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = load_model(checkpoint_path, ...)

    # # Load ligand library
    # ligand_library = load_ligand_library(ligand_library_path)

    # # Process all graphs in directory
    # graph_dir = Path(graph_dir)
    # results = {}

    # for graph_path in tqdm(list(graph_dir.glob("*.pt"))):
    #     complex_id = graph_path.stem
    #     graph = torch.load(graph_path)
    #
    #     # Predict embedding
    #     embedding = predict_pocket_embedding(graph, model, device)
    #
    #     # Find similar ligands
    #     similar = find_similar_ligands(embedding, ligand_library, top_k=10)
    #
    #     results[complex_id] = similar

    # # Save results
    # with open(output_dir / "batch_results.json", 'w') as f:
    #     json.dump(results, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print("RNA Pocket Encoder - Inference Script")
        print("=" * 60)
        print("\nUsage:")
        print("  python 05_run_inference.py --checkpoint <path> --query_graph <path> --ligand_library <path>")
        print("\nExample:")
        print("  python 05_run_inference.py \\")
        print("    --checkpoint models/checkpoints/best_model.pt \\")
        print("    --query_graph data/processed/graphs/1abc_LIG.pt \\")
        print("    --ligand_library data/processed/ligand_embeddings.h5 \\")
        print("    --top_k 10 \\")
        print("    --output results/predictions.json")
        print("\nFor more options, use --help")
