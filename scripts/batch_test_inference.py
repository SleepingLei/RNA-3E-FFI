#!/usr/bin/env python3
"""
Batch Inference on Test Set with Ranking Statistics

This script:
1. Loads all test samples from splits.json
2. Runs inference on each sample
3. Finds the rank of the true ligand in predictions
4. Computes statistics (MRR, Top-K accuracy, etc.)
"""
import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import h5py
from tqdm import tqdm
from collections import defaultdict

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.e3_gnn_encoder import RNAPocketEncoder


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    model = RNAPocketEncoder(
        input_dim=config['input_dim'],
        hidden_irreps=config['hidden_irreps'],
        output_dim=config['output_dim'],
        num_layers=config['num_layers'],
        num_radial_basis=config['num_radial_basis']
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    return model


def predict_pocket_embedding(pocket_graph, model, device):
    """Predict embedding for a single RNA pocket."""
    model.eval()

    with torch.no_grad():
        pocket_graph = pocket_graph.to(device)

        if not hasattr(pocket_graph, 'batch') or pocket_graph.batch is None:
            pocket_graph.batch = torch.zeros(pocket_graph.num_nodes, dtype=torch.long, device=device)

        embedding = model(pocket_graph)
        embedding_np = embedding.cpu().numpy()

        if len(embedding_np.shape) > 1 and embedding_np.shape[0] == 1:
            embedding_np = embedding_np[0]

    return embedding_np


def calculate_distance(embedding1, embedding2, metric='euclidean'):
    """Calculate distance between two embeddings."""
    if metric == 'euclidean':
        return np.linalg.norm(embedding1 - embedding2)
    elif metric == 'cosine':
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        cosine_sim = dot_product / (norm1 * norm2 + 1e-8)
        return 1 - cosine_sim
    else:
        raise ValueError(f"Unknown metric: {metric}")


def find_ligand_rank(query_embedding, ligand_library, true_ligand_id, metric='euclidean'):
    """
    Find the rank of the true ligand in predictions.

    Args:
        query_embedding: Query pocket embedding
        ligand_library: Dictionary mapping ligand IDs to embeddings
        true_ligand_id: The true ligand ID for this pocket
        metric: Distance metric to use

    Returns:
        rank: Rank of true ligand (1-indexed, 1 is best)
        all_distances: List of (ligand_id, distance) sorted by distance
    """
    distances = []

    for ligand_id, ligand_embedding in ligand_library.items():
        dist = calculate_distance(query_embedding, ligand_embedding, metric)
        distances.append((ligand_id, dist))

    # Sort by distance (ascending - closer is better)
    distances.sort(key=lambda x: x[1])

    # Find rank of true ligand
    rank = None
    for i, (ligand_id, dist) in enumerate(distances, 1):
        if ligand_id == true_ligand_id:
            rank = i
            break

    return rank, distances


def load_ligand_library(embeddings_path):
    """Load pre-computed ligand embeddings."""
    print(f"Loading ligand library from {embeddings_path}...")

    ligand_library = {}
    with h5py.File(embeddings_path, 'r') as f:
        for ligand_id in tqdm(f.keys(), desc="Loading embeddings"):
            ligand_library[ligand_id] = np.array(f[ligand_id][:])

    print(f"Loaded {len(ligand_library)} ligand embeddings")
    return ligand_library


def extract_ligand_from_sample_name(sample_name):
    """
    Extract ligand ID from sample name.
    E.g., '2kx8_ARG_model2' -> 'ARG'
         '1aju_ARG_model10' -> 'ARG'
    """
    parts = sample_name.split('_')
    # Format: pdbid_ligandid_modelN
    # Remove 'modelN' part and pdbid part
    if len(parts) >= 3 and parts[-1].startswith('model'):
        ligand_id = '_'.join(parts[1:-1])  # Handle multi-part ligands
    else:
        ligand_id = '_'.join(parts[1:])

    return ligand_id


def get_graph_path_from_sample(sample_name, graphs_dir):
    """
    Get graph file path from sample name.
    E.g., '2kx8_ARG_model2' -> 'data/processed/graphs/2kx8_ARG_model2.pt'
    """
    # Keep the full sample name including model suffix
    return Path(graphs_dir) / f"{sample_name}.pt"


def compute_metrics(ranks):
    """
    Compute ranking metrics.

    Args:
        ranks: List of ranks (1-indexed)

    Returns:
        Dictionary of metrics
    """
    ranks = np.array(ranks)

    metrics = {
        'total_samples': len(ranks),
        'mean_rank': float(np.mean(ranks)),
        'median_rank': float(np.median(ranks)),
        'mrr': float(np.mean(1.0 / ranks)),  # Mean Reciprocal Rank
        'top1_accuracy': float(np.mean(ranks == 1)),
        'top3_accuracy': float(np.mean(ranks <= 3)),
        'top5_accuracy': float(np.mean(ranks <= 5)),
        'top10_accuracy': float(np.mean(ranks <= 10)),
        'top20_accuracy': float(np.mean(ranks <= 20)),
        'top50_accuracy': float(np.mean(ranks <= 50)),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Batch inference on test set")

    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config JSON")

    # Data arguments
    parser.add_argument("--splits", type=str, default="data/splits/splits.json",
                        help="Path to splits JSON file")
    parser.add_argument("--graphs_dir", type=str, default="data/processed/graphs",
                        help="Directory containing pocket graph files")
    parser.add_argument("--ligand_library", type=str, required=True,
                        help="Path to ligand embeddings HDF5 file")

    # Inference arguments
    parser.add_argument("--metric", type=str, default='euclidean',
                        choices=['euclidean', 'cosine'],
                        help="Distance metric to use")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--save_top_k", type=int, default=50,
                        help="Save top K predictions for each sample")

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    print(f"\nLoading config from {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Load ligand library
    ligand_library = load_ligand_library(args.ligand_library)

    # Load test samples
    print(f"\nLoading test samples from {args.splits}")
    with open(args.splits, 'r') as f:
        splits = json.load(f)
    test_samples = splits['test']
    print(f"Found {len(test_samples)} test samples")

    # Run inference on all test samples
    results = []
    ranks = []
    missing_graphs = []
    missing_ligands = []

    print(f"\nRunning inference on test samples...")
    for sample_name in tqdm(test_samples, desc="Processing"):
        # Get graph path
        graph_path = get_graph_path_from_sample(sample_name, args.graphs_dir)

        # Check if graph exists
        if not graph_path.exists():
            missing_graphs.append(sample_name)
            continue

        # Extract true ligand
        true_ligand_id = extract_ligand_from_sample_name(sample_name)

        # Check if true ligand is in library
        if true_ligand_id not in ligand_library:
            missing_ligands.append((sample_name, true_ligand_id))
            continue

        # Load graph
        try:
            query_graph = torch.load(graph_path)
        except Exception as e:
            print(f"\nError loading {graph_path}: {e}")
            continue

        # Predict embedding
        query_embedding = predict_pocket_embedding(query_graph, model, device)

        # Find rank
        rank, all_distances = find_ligand_rank(
            query_embedding,
            ligand_library,
            true_ligand_id,
            metric=args.metric
        )

        ranks.append(rank)

        # Store result
        result = {
            'sample_name': sample_name,
            'graph_path': str(graph_path),
            'true_ligand': true_ligand_id,
            'rank': rank,
            'top_predictions': [
                {'ligand_id': lid, 'distance': float(dist)}
                for lid, dist in all_distances[:args.save_top_k]
            ]
        }
        results.append(result)

    # Compute metrics
    print(f"\n{'='*60}")
    print("RANKING STATISTICS")
    print(f"{'='*60}")

    if ranks:
        metrics = compute_metrics(ranks)

        print(f"\nTotal test samples processed: {metrics['total_samples']}")
        print(f"\nRanking Metrics:")
        print(f"  Mean Rank:        {metrics['mean_rank']:.2f}")
        print(f"  Median Rank:      {metrics['median_rank']:.1f}")
        print(f"  MRR:              {metrics['mrr']:.4f}")
        print(f"\nTop-K Accuracy:")
        print(f"  Top-1:            {metrics['top1_accuracy']*100:.2f}%")
        print(f"  Top-3:            {metrics['top3_accuracy']*100:.2f}%")
        print(f"  Top-5:            {metrics['top5_accuracy']*100:.2f}%")
        print(f"  Top-10:           {metrics['top10_accuracy']*100:.2f}%")
        print(f"  Top-20:           {metrics['top20_accuracy']*100:.2f}%")
        print(f"  Top-50:           {metrics['top50_accuracy']*100:.2f}%")

        # Show worst performing samples
        worst_samples = sorted(results, key=lambda x: x['rank'], reverse=True)[:10]
        print(f"\n{'='*60}")
        print("Worst 10 Predictions:")
        print(f"{'='*60}")
        for i, sample in enumerate(worst_samples, 1):
            print(f"{i}. {sample['sample_name']:30s} Rank: {sample['rank']:4d}  True Ligand: {sample['true_ligand']}")

        # Show best performing samples (rank = 1)
        best_samples = [r for r in results if r['rank'] == 1]
        print(f"\n{'='*60}")
        print(f"Perfect Predictions (Rank = 1): {len(best_samples)}")
        print(f"{'='*60}")

    # Report missing data
    if missing_graphs:
        print(f"\n{'='*60}")
        print(f"WARNING: {len(missing_graphs)} samples have missing graph files:")
        for sample in missing_graphs[:10]:
            print(f"  - {sample}")
        if len(missing_graphs) > 10:
            print(f"  ... and {len(missing_graphs) - 10} more")

    if missing_ligands:
        print(f"\n{'='*60}")
        print(f"WARNING: {len(missing_ligands)} samples have ligands not in library:")
        for sample, ligand in missing_ligands[:10]:
            print(f"  - {sample} (ligand: {ligand})")
        if len(missing_ligands) > 10:
            print(f"  ... and {len(missing_ligands) - 10} more")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_file = output_dir / "test_results_detailed.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': config,
            'metrics': metrics if ranks else {},
            'results': results,
            'missing_graphs': missing_graphs,
            'missing_ligands': [{'sample': s, 'ligand': l} for s, l in missing_ligands]
        }, f, indent=2)
    print(f"\nDetailed results saved to {results_file}")

    # Save summary metrics
    summary_file = output_dir / "test_metrics_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(metrics if ranks else {}, f, indent=2)
    print(f"Summary metrics saved to {summary_file}")

    # Save ranks only (for easy analysis)
    ranks_file = output_dir / "test_ranks.txt"
    with open(ranks_file, 'w') as f:
        f.write("sample_name\ttrue_ligand\trank\n")
        for result in results:
            f.write(f"{result['sample_name']}\t{result['true_ligand']}\t{result['rank']}\n")
    print(f"Ranks saved to {ranks_file}")

    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
