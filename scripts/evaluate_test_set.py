#!/usr/bin/env python3
"""
Test Set Evaluation Script

This script evaluates the trained model on the test set and computes hit rate metrics
at different thresholds (top 5%, 10%, 20%).

Hit rate is calculated as: for each test sample, whether the correct ligand appears
in the top-k% of retrieved candidates.
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


def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """
    Load model from checkpoint.

    Automatically detects model version and architecture from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model in eval mode and its configuration
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

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
        warnings.warn("Checkpoint does not contain model configuration and no config.json found. "
                     "Using default v2 configuration.")

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
            output_dim=config.get('output_dim', 1536),
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
            output_dim=config.get('output_dim', 1536),
            num_layers=config.get('num_layers', 4),
            num_radial_basis=config.get('num_radial_basis', 8),
            use_multi_hop=config.get('use_multi_hop', True),
            use_nonbonded=config.get('use_nonbonded', True),
            use_gate=config.get('use_gate', True),
            use_layer_norm=config.get('use_layer_norm', False),
            pooling_type=config.get('pooling_type', 'attention'),
            dropout=config.get('dropout', 0.0)
        )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, config


def load_test_split(splits_path):
    """
    Load test split from splits.json.

    Args:
        splits_path: Path to splits.json

    Returns:
        List of test complex IDs
    """
    print(f"\nLoading test split from {splits_path}...")

    with open(splits_path, 'r') as f:
        splits = json.load(f)

    test_ids = splits['test']
    print(f"✓ Loaded {len(test_ids)} test samples")

    return test_ids


def load_ligand_library(embeddings_path):
    """
    Load pre-computed ligand embeddings.

    Args:
        embeddings_path: Path to HDF5 file with ligand embeddings
                        (should be deduplicated with ligand names as keys)

    Returns:
        Dictionary mapping ligand names (e.g., "ARG", "SAM") to embeddings
    """
    print(f"\nLoading ligand library from {embeddings_path}...")

    ligand_library = {}
    with h5py.File(embeddings_path, 'r') as f:
        for ligand_id in tqdm(f.keys(), desc="Loading embeddings"):
            ligand_library[ligand_id] = np.array(f[ligand_id][:])

    print(f"✓ Loaded {len(ligand_library)} ligand embeddings")
    return ligand_library


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
        # Return the ligand name (everything after first underscore)
        return '_'.join(parts[1:])
    else:
        # Fallback: return as-is
        return base


def predict_embedding(model, graph, device):
    """
    Predict pocket embedding for a graph.

    Args:
        model: Trained model
        graph: PyTorch Geometric Data object
        device: Device to run on

    Returns:
        Predicted embedding as numpy array
    """
    model.eval()

    with torch.no_grad():
        graph = graph.to(device)

        # Ensure batch attribute exists
        if not hasattr(graph, 'batch') or graph.batch is None:
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)

        # Forward pass
        embedding = model(graph)

        # Convert to numpy
        embedding_np = embedding.cpu().numpy()

        # Handle batch dimension
        if len(embedding_np.shape) > 1 and embedding_np.shape[0] == 1:
            embedding_np = embedding_np[0]

    return embedding_np


def calculate_distance(embedding1, embedding2, metric='cosine'):
    """
    Calculate distance between two embeddings.

    Args:
        embedding1: First embedding
        embedding2: Second embedding
        metric: 'cosine' or 'euclidean'

    Returns:
        Distance value
    """
    if metric == 'cosine':
        # Cosine distance = 1 - cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        cosine_sim = dot_product / (norm1 * norm2 + 1e-8)
        return 1 - cosine_sim
    elif metric == 'euclidean':
        return np.linalg.norm(embedding1 - embedding2)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def retrieve_top_k(query_embedding, ligand_library, k, metric='cosine'):
    """
    Retrieve top-k most similar ligands.

    Args:
        query_embedding: Query pocket embedding
        ligand_library: Dictionary of ligand embeddings
        k: Number of top matches to retrieve
        metric: Distance metric

    Returns:
        List of (ligand_id, distance) tuples
    """
    distances = []

    for ligand_id, ligand_embedding in ligand_library.items():
        dist = calculate_distance(query_embedding, ligand_embedding, metric)
        distances.append((ligand_id, dist))

    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[1])

    return distances[:k]


def evaluate_test_set(model, test_ids, graph_dir, ligand_library, device,
                     metric='cosine', top_percentages=[5, 10, 20]):
    """
    Evaluate model on test set and compute hit rates.

    Args:
        model: Trained model
        test_ids: List of test complex IDs (e.g., "2kx8_ARG_model2")
        graph_dir: Directory containing graph files
        ligand_library: Dictionary of ligand embeddings with ligand names as keys
                       (e.g., {"ARG": embedding, "SAM": embedding, ...})
        device: Device to run on
        metric: Distance metric for retrieval
        top_percentages: List of percentage thresholds for hit rate

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*60}")
    print("Evaluating Test Set")
    print(f"{'='*60}\n")

    total_ligands = len(ligand_library)
    print(f"Total ligands in library: {total_ligands}")
    print(f"Distance metric: {metric}")
    print(f"Hit rate thresholds: {top_percentages}%")

    # Calculate k values for each percentage
    k_values = {pct: max(1, int(total_ligands * pct / 100)) for pct in top_percentages}
    print(f"Top-k values: {k_values}")

    results = {
        'total_samples': len(test_ids),
        'successful_predictions': 0,
        'failed_predictions': [],
        'metric': metric,
        'total_ligands': total_ligands,
        'hit_rates': {f'top{pct}%': {'hits': 0, 'misses': 0} for pct in top_percentages},
        'detailed_results': []
    }

    # Track distance statistics
    correct_ligand_ranks = []
    correct_ligand_distances = []

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

        # Get ground truth ligand name
        true_ligand_id = extract_ligand_name(complex_id)

        if true_ligand_id not in ligand_library:
            results['failed_predictions'].append({
                'complex_id': complex_id,
                'reason': 'ligand_not_in_library'
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

        # Calculate distances to all ligands and find rank of correct ligand
        all_distances = []
        for ligand_id, ligand_embedding in ligand_library.items():
            dist = calculate_distance(predicted_embedding, ligand_embedding, metric)
            all_distances.append((ligand_id, dist))

        # Sort by distance
        all_distances.sort(key=lambda x: x[1])

        # Find rank of correct ligand (1-indexed)
        rank = None
        correct_distance = None
        for i, (ligand_id, dist) in enumerate(all_distances, 1):
            if ligand_id == true_ligand_id:
                rank = i
                correct_distance = dist
                break

        if rank is None:
            results['failed_predictions'].append({
                'complex_id': complex_id,
                'reason': 'correct_ligand_not_ranked'
            })
            continue

        # Update statistics
        results['successful_predictions'] += 1
        correct_ligand_ranks.append(rank)
        correct_ligand_distances.append(correct_distance)

        # Check hit rates at different thresholds
        sample_result = {
            'complex_id': complex_id,
            'true_ligand_id': true_ligand_id,
            'rank': rank,
            'distance': float(correct_distance),
            'hits': {}
        }

        for pct in top_percentages:
            k = k_values[pct]
            is_hit = rank <= k

            if is_hit:
                results['hit_rates'][f'top{pct}%']['hits'] += 1
            else:
                results['hit_rates'][f'top{pct}%']['misses'] += 1

            sample_result['hits'][f'top{pct}%'] = is_hit

        results['detailed_results'].append(sample_result)

    # Compute final hit rate percentages
    n_successful = results['successful_predictions']

    if n_successful > 0:
        for pct in top_percentages:
            key = f'top{pct}%'
            hits = results['hit_rates'][key]['hits']
            hit_rate = hits / n_successful * 100
            results['hit_rates'][key]['hit_rate'] = hit_rate
            results['hit_rates'][key]['k_value'] = k_values[pct]

        # Add rank statistics
        results['rank_statistics'] = {
            'mean_rank': float(np.mean(correct_ligand_ranks)),
            'median_rank': float(np.median(correct_ligand_ranks)),
            'min_rank': int(np.min(correct_ligand_ranks)),
            'max_rank': int(np.max(correct_ligand_ranks)),
            'std_rank': float(np.std(correct_ligand_ranks))
        }

        # Add distance statistics
        results['distance_statistics'] = {
            'mean_distance': float(np.mean(correct_ligand_distances)),
            'median_distance': float(np.median(correct_ligand_distances)),
            'min_distance': float(np.min(correct_ligand_distances)),
            'max_distance': float(np.max(correct_ligand_distances)),
            'std_distance': float(np.std(correct_ligand_distances))
        }

    return results


def print_results(results):
    """Print evaluation results in a nice format."""
    print(f"\n{'='*60}")
    print("Test Set Evaluation Results")
    print(f"{'='*60}\n")

    print(f"Total samples:           {results['total_samples']}")
    print(f"Successful predictions:  {results['successful_predictions']}")
    print(f"Failed predictions:      {len(results['failed_predictions'])}")
    print(f"Total ligands in library: {results['total_ligands']}")
    print(f"Distance metric:         {results['metric']}")

    if results['failed_predictions']:
        print(f"\n⚠️  Failed predictions:")
        failure_reasons = defaultdict(int)
        for failure in results['failed_predictions']:
            failure_reasons[failure['reason']] += 1
        for reason, count in failure_reasons.items():
            print(f"  - {reason}: {count}")

    print(f"\n{'='*60}")
    print("Hit Rates")
    print(f"{'='*60}")

    for key in sorted(results['hit_rates'].keys()):
        data = results['hit_rates'][key]
        hit_rate = data.get('hit_rate', 0.0)
        k_value = data.get('k_value', 0)
        hits = data['hits']
        total = hits + data['misses']

        print(f"\n{key:>8s} (k={k_value}):")
        print(f"  Hit rate: {hit_rate:6.2f}% ({hits}/{total})")

    if 'rank_statistics' in results:
        print(f"\n{'='*60}")
        print("Rank Statistics (of correct ligand)")
        print(f"{'='*60}")

        stats = results['rank_statistics']
        print(f"  Mean rank:   {stats['mean_rank']:.2f}")
        print(f"  Median rank: {stats['median_rank']:.1f}")
        print(f"  Min rank:    {stats['min_rank']}")
        print(f"  Max rank:    {stats['max_rank']}")
        print(f"  Std rank:    {stats['std_rank']:.2f}")

    if 'distance_statistics' in results:
        print(f"\n{'='*60}")
        print("Distance Statistics (to correct ligand)")
        print(f"{'='*60}")

        stats = results['distance_statistics']
        print(f"  Mean distance:   {stats['mean_distance']:.6f}")
        print(f"  Median distance: {stats['median_distance']:.6f}")
        print(f"  Min distance:    {stats['min_distance']:.6f}")
        print(f"  Max distance:    {stats['max_distance']:.6f}")
        print(f"  Std distance:    {stats['std_distance']:.6f}")

    print(f"\n{'='*60}\n")


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on test set",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--splits", type=str, default="data/splits/filtered_splits.json",
                        help="Path to splits.json file")
    parser.add_argument("--graph_dir", type=str, default="data/processed/graphs",
                        help="Directory containing graph files")
    parser.add_argument("--ligand_embeddings", type=str,
                        default="data/processed/ligand_embeddings_dedup.h5",
                        help="Path to deduplicated ligand embeddings HDF5 file (with ligand names as keys)")

    # Optional arguments
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--metric", type=str, default="euclidean",
                        choices=['cosine', 'euclidean'],
                        help="Distance metric for retrieval")
    parser.add_argument("--top_percentages", type=int, nargs='+', default=[5, 10, 20],
                        help="Percentage thresholds for hit rate (default: 5 10 20)")
    parser.add_argument("--use_normalization", action="store_true",
                        help="Apply normalization using saved parameters")
    parser.add_argument("--norm_params_dir", type=str, default="data/processed",
                        help="Directory containing normalization parameter files")

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, config = load_model_from_checkpoint(args.checkpoint, device)

    # Load test split
    test_ids = load_test_split(args.splits)

    # Load ligand library
    ligand_library = load_ligand_library(args.ligand_embeddings)

    # Optionally apply normalization
    if args.use_normalization:
        print(f"\n⚠️  Normalization is enabled but not yet implemented in this script.")
        print(f"Normalization should be applied during data preprocessing, not during inference.")
        print(f"Make sure your graphs and embeddings were normalized during training!")

    # Run evaluation
    results = evaluate_test_set(
        model=model,
        test_ids=test_ids,
        graph_dir=args.graph_dir,
        ligand_library=ligand_library,
        device=device,
        metric=args.metric,
        top_percentages=args.top_percentages
    )

    # Print results
    print_results(results)

    # Save results to file
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
