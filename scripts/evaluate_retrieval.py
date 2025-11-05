#!/usr/bin/env python3
"""
Retrieval Evaluation Script

This script evaluates the trained model on retrieval task:
- Given a bait library of ligand embeddings
- For each RNA pocket, find the true ligand from the bait library
- Compute normalized rank (0-1, lower is better)
- Report various retrieval metrics: MRR, Recall@K, Mean Normalized Rank

Key metrics:
- Normalized Rank: (rank - 1) / (total_baits - 1), range [0, 1]
  * 0 = best (rank 1)
  * 1 = worst (rank = total_baits)
- Mean Reciprocal Rank (MRR): 1 / rank, averaged
- Recall@K: percentage of samples where true ligand is in top-K
- Top-1 Accuracy: percentage where rank = 1
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
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config from checkpoint or config.json
    config = checkpoint.get('config', {})

    checkpoint_dir = Path(checkpoint_path).parent
    config_json_path = checkpoint_dir / 'config.json'

    if config_json_path.exists():
        print(f"  Loading config from {config_json_path}...")
        with open(config_json_path, 'r') as f:
            file_config = json.load(f)
            for key in ['atom_embed_dim', 'residue_embed_dim', 'hidden_irreps',
                       'output_dim', 'num_layers', 'use_multi_hop', 'use_nonbonded',
                       'pooling_type', 'use_layer_norm', 'dropout']:
                if key in file_config:
                    config[key] = file_config[key]

    model_version = config.get('model_version', 'v2')

    print(f"  Model version: {model_version}")
    print(f"  Training epoch: {checkpoint.get('epoch', 'unknown')}")

    # Import model
    from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2
    from scripts.amber_vocabulary import get_global_encoder

    encoder = get_global_encoder()

    model = RNAPocketEncoderV2(
        num_atom_types=encoder.num_atom_types,
        num_residues=encoder.num_residues,
        atom_embed_dim=config.get('atom_embed_dim', 32),
        residue_embed_dim=config.get('residue_embed_dim', 16),
        hidden_irreps=config.get('hidden_irreps', '32x0e + 16x1o + 8x2e'),
        output_dim=config.get('output_dim', 512),
        num_layers=config.get('num_layers', 4),
        use_multi_hop=config.get('use_multi_hop', True),
        use_nonbonded=config.get('use_nonbonded', True),
        use_layer_norm=config.get('use_layer_norm', False),
        pooling_type=config.get('pooling_type', 'attention'),
        dropout=config.get('dropout', 0.0)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    return model, config


def load_test_split(splits_path):
    """
    Load test split from splits.json.

    Returns:
        List of test complex IDs
    """
    print(f"\nLoading test split from {splits_path}...")

    with open(splits_path, 'r') as f:
        splits = json.load(f)

    test_ids = splits['test']
    print(f"✓ Loaded {len(test_ids)} test samples")

    return test_ids


def load_bait_library(embeddings_path):
    """
    Load bait library ligand embeddings.

    Args:
        embeddings_path: Path to HDF5 file with bait ligand embeddings

    Returns:
        Dictionary mapping ligand IDs to embeddings
    """
    print(f"\nLoading bait library from {embeddings_path}...")

    bait_library = {}
    with h5py.File(embeddings_path, 'r') as f:
        for ligand_id in tqdm(f.keys(), desc="Loading bait embeddings"):
            bait_library[ligand_id] = np.array(f[ligand_id][:])

    print(f"✓ Loaded {len(bait_library)} bait ligand embeddings")
    return bait_library


def extract_ligand_name(complex_id):
    """
    Extract ligand name from complex ID.

    Args:
        complex_id: e.g., "2kx8_ARG_model2" or "1uui_P12_model0"

    Returns:
        Ligand name only, e.g., "ARG" or "P12"
    """
    # Remove model suffix if present
    if '_model' in complex_id:
        base = complex_id.split('_model')[0]
    else:
        base = complex_id

    # Split by underscore and extract ligand part
    # Format: <pdb_id>_<ligand_name>
    parts = base.split('_')

    if len(parts) >= 2:
        return '_'.join(parts[1:])
    else:
        return base


def predict_embedding(model, graph, device):
    """
    Predict pocket embedding for a graph.

    Returns:
        Predicted embedding as numpy array
    """
    model.eval()

    with torch.no_grad():
        graph = graph.to(device)

        if not hasattr(graph, 'batch') or graph.batch is None:
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)

        embedding = model(graph)
        embedding_np = embedding.cpu().numpy()

        if len(embedding_np.shape) > 1 and embedding_np.shape[0] == 1:
            embedding_np = embedding_np[0]

    return embedding_np


def calculate_similarity(embedding1, embedding2, metric='cosine'):
    """
    Calculate similarity between two embeddings.

    Args:
        embedding1: First embedding
        embedding2: Second embedding
        metric: 'cosine' or 'euclidean'

    Returns:
        Similarity value (higher is better for retrieval)
    """
    if metric == 'cosine':
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2 + 1e-8)
    elif metric == 'euclidean':
        # Negative distance (so higher is better)
        return -np.linalg.norm(embedding1 - embedding2)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def evaluate_retrieval(model, test_ids, graph_dir, bait_library, device,
                      metric='cosine', recall_k_values=[1, 5, 10, 20, 50]):
    """
    Evaluate retrieval performance.

    Args:
        model: Trained model
        test_ids: List of test complex IDs
        graph_dir: Directory containing graph files
        bait_library: Dictionary of bait ligand embeddings
        device: Device to run on
        metric: Similarity metric ('cosine' or 'euclidean')
        recall_k_values: List of K values for Recall@K

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*70}")
    print("Retrieval Evaluation")
    print(f"{'='*70}\n")

    total_baits = len(bait_library)
    print(f"Total baits in library: {total_baits}")
    print(f"Similarity metric: {metric}")
    print(f"Recall@K values: {recall_k_values}")

    results = {
        'total_samples': len(test_ids),
        'successful_predictions': 0,
        'failed_predictions': [],
        'metric': metric,
        'total_baits': total_baits,
        'detailed_results': []
    }

    # Track metrics
    ranks = []
    normalized_ranks = []
    reciprocal_ranks = []
    recall_hits = {k: 0 for k in recall_k_values}

    print(f"\nProcessing {len(test_ids)} test samples...")

    for complex_id in tqdm(test_ids, desc="Evaluating"):
        # Load graph
        graph_path = Path(graph_dir) / f"{complex_id}.pt"

        if not graph_path.exists():
            results['failed_predictions'].append({
                'complex_id': complex_id,
                'reason': 'graph_file_not_found'
            })
            continue

        try:
            graph = torch.load(graph_path, weights_only=False)
        except Exception as e:
            results['failed_predictions'].append({
                'complex_id': complex_id,
                'reason': f'graph_loading_error: {str(e)}'
            })
            continue

        # Get ground truth ligand
        true_ligand_id = extract_ligand_name(complex_id)

        if true_ligand_id not in bait_library:
            results['failed_predictions'].append({
                'complex_id': complex_id,
                'reason': f'ligand_not_in_bait_library (ligand={true_ligand_id})'
            })
            continue

        # Predict embedding
        try:
            predicted_embedding = predict_embedding(model, graph, device)
        except Exception as e:
            results['failed_predictions'].append({
                'complex_id': complex_id,
                'reason': f'prediction_error: {str(e)}'
            })
            continue

        # Calculate similarities to all baits
        similarities = []
        for bait_id, bait_embedding in bait_library.items():
            sim = calculate_similarity(predicted_embedding, bait_embedding, metric)
            similarities.append((bait_id, sim))

        # Sort by similarity (descending, higher is better)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Find rank of true ligand (1-indexed)
        rank = None
        true_similarity = None
        for i, (bait_id, sim) in enumerate(similarities, 1):
            if bait_id == true_ligand_id:
                rank = i
                true_similarity = sim
                break

        if rank is None:
            results['failed_predictions'].append({
                'complex_id': complex_id,
                'reason': 'true_ligand_not_ranked'
            })
            continue

        # Calculate metrics
        # Normalized rank: (rank - 1) / (total - 1)
        # Range: [0, 1], where 0 is best (rank=1) and 1 is worst (rank=total)
        normalized_rank = (rank - 1) / (total_baits - 1) if total_baits > 1 else 0.0

        # Reciprocal rank: 1 / rank
        reciprocal_rank = 1.0 / rank

        # Update statistics
        results['successful_predictions'] += 1
        ranks.append(rank)
        normalized_ranks.append(normalized_rank)
        reciprocal_ranks.append(reciprocal_rank)

        # Check Recall@K
        recall_status = {}
        for k in recall_k_values:
            is_hit = rank <= k
            if is_hit:
                recall_hits[k] += 1
            recall_status[f'recall@{k}'] = is_hit

        # Store detailed result
        sample_result = {
            'complex_id': complex_id,
            'true_ligand_id': true_ligand_id,
            'rank': rank,
            'normalized_rank': float(normalized_rank),
            'reciprocal_rank': float(reciprocal_rank),
            'similarity': float(true_similarity),
            'top_5_predictions': [
                {'ligand_id': lid, 'similarity': float(sim)}
                for lid, sim in similarities[:5]
            ],
            'recall': recall_status
        }

        results['detailed_results'].append(sample_result)

    # Compute final metrics
    n_successful = results['successful_predictions']

    if n_successful > 0:
        # Rank statistics
        results['metrics'] = {
            'mean_rank': float(np.mean(ranks)),
            'median_rank': float(np.median(ranks)),
            'min_rank': int(np.min(ranks)),
            'max_rank': int(np.max(ranks)),
            'std_rank': float(np.std(ranks)),

            # Normalized rank (0-1, lower is better)
            'mean_normalized_rank': float(np.mean(normalized_ranks)),
            'median_normalized_rank': float(np.median(normalized_ranks)),
            'min_normalized_rank': float(np.min(normalized_ranks)),
            'max_normalized_rank': float(np.max(normalized_ranks)),
            'std_normalized_rank': float(np.std(normalized_ranks)),

            # Mean Reciprocal Rank (higher is better)
            'mrr': float(np.mean(reciprocal_ranks)),

            # Top-1 Accuracy
            'top1_accuracy': float(sum(1 for r in ranks if r == 1) / n_successful * 100),

            # Recall@K
            'recall': {}
        }

        for k in recall_k_values:
            recall_rate = recall_hits[k] / n_successful * 100
            results['metrics']['recall'][f'recall@{k}'] = {
                'hits': recall_hits[k],
                'total': n_successful,
                'percentage': float(recall_rate)
            }

    return results


def print_results(results):
    """Print evaluation results in a nice format."""
    print(f"\n{'='*70}")
    print("Retrieval Evaluation Results")
    print(f"{'='*70}\n")

    print(f"Total samples:           {results['total_samples']}")
    print(f"Successful predictions:  {results['successful_predictions']}")
    print(f"Failed predictions:      {len(results['failed_predictions'])}")
    print(f"Total baits in library:  {results['total_baits']}")
    print(f"Similarity metric:       {results['metric']}")

    if results['failed_predictions']:
        print(f"\n⚠️  Failed predictions:")
        failure_reasons = defaultdict(int)
        for failure in results['failed_predictions']:
            failure_reasons[failure['reason']] += 1
        for reason, count in failure_reasons.items():
            print(f"  - {reason}: {count}")

    if 'metrics' in results:
        metrics = results['metrics']

        print(f"\n{'='*70}")
        print("Key Metrics")
        print(f"{'='*70}")

        print(f"\n📊 Normalized Rank (0-1, lower is better):")
        print(f"  Mean:   {metrics['mean_normalized_rank']:.4f}")
        print(f"  Median: {metrics['median_normalized_rank']:.4f}")
        print(f"  Std:    {metrics['std_normalized_rank']:.4f}")
        print(f"  Range:  [{metrics['min_normalized_rank']:.4f}, {metrics['max_normalized_rank']:.4f}]")

        print(f"\n🎯 Mean Reciprocal Rank (MRR, higher is better):")
        print(f"  MRR: {metrics['mrr']:.4f}")

        print(f"\n🏆 Top-1 Accuracy:")
        print(f"  {metrics['top1_accuracy']:.2f}%")

        print(f"\n{'='*70}")
        print("Rank Statistics (absolute)")
        print(f"{'='*70}")

        print(f"  Mean rank:   {metrics['mean_rank']:.2f}")
        print(f"  Median rank: {metrics['median_rank']:.1f}")
        print(f"  Min rank:    {metrics['min_rank']}")
        print(f"  Max rank:    {metrics['max_rank']}")
        print(f"  Std rank:    {metrics['std_rank']:.2f}")

        print(f"\n{'='*70}")
        print("Recall@K")
        print(f"{'='*70}")

        for key in sorted(results['metrics']['recall'].keys()):
            data = results['metrics']['recall'][key]
            k = key.split('@')[1]
            percentage = data['percentage']
            hits = data['hits']
            total = data['total']

            print(f"\n  {key:>12s}: {percentage:6.2f}% ({hits}/{total})")

    print(f"\n{'='*70}\n")


def save_results(results, output_path):
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to {output_path}")


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval task: find true ligand from bait library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/evaluate_retrieval.py \\
    --checkpoint models/best_model.pt \\
    --bait_library data/processed/ligand_embeddings_dedup.h5

  # With custom recall thresholds
  python scripts/evaluate_retrieval.py \\
    --checkpoint models/best_model.pt \\
    --bait_library data/bait_library.h5 \\
    --recall_k 1 5 10 20 50 100 \\
    --output results/retrieval_results.json
        """
    )

    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--bait_library", type=str, required=True,
                        help="Path to bait library HDF5 file (ligand embeddings)")

    # Optional arguments
    parser.add_argument("--splits", type=str, default="data/splits/splits.json",
                        help="Path to splits.json file (default: data/splits/splits.json)")
    parser.add_argument("--graph_dir", type=str, default="data/processed/graphs",
                        help="Directory containing graph files (default: data/processed/graphs)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results (default: None, no save)")
    parser.add_argument("--metric", type=str, default="cosine",
                        choices=['cosine', 'euclidean'],
                        help="Similarity metric for retrieval (default: cosine)")
    parser.add_argument("--recall_k", type=int, nargs='+', default=[1, 5, 10, 20, 50],
                        help="K values for Recall@K (default: 1 5 10 20 50)")

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, config = load_model_from_checkpoint(args.checkpoint, device)

    # Load test split
    test_ids = load_test_split(args.splits)

    # Load bait library
    bait_library = load_bait_library(args.bait_library)

    # Run evaluation
    results = evaluate_retrieval(
        model=model,
        test_ids=test_ids,
        graph_dir=args.graph_dir,
        bait_library=bait_library,
        device=device,
        metric=args.metric,
        recall_k_values=args.recall_k
    )

    # Print results
    print_results(results)

    # Save results to file
    if args.output:
        save_results(results, args.output)

    return results


if __name__ == "__main__":
    main()
