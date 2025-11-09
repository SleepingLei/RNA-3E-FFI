#!/usr/bin/env python3
"""
Sample Pre-filtering Script

This script analyzes all graph samples and filters out overly large ones
that may cause GPU Out-of-Memory errors during training.

Features:
- Statistics on graph sizes (nodes, edges, file size)
- Automatic filtering based on thresholds
- Generates filtered dataset splits
- Saves detailed analysis report

Usage:
    python scripts/filter_large_samples.py --graph_dir data/processed/graphs
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_graph_statistics(graph_path):
    """
    Extract statistics from a graph file.

    Args:
        graph_path: Path to .pt graph file

    Returns:
        Dictionary with graph statistics
    """
    try:
        # Load graph
        data = torch.load(graph_path, weights_only=False)

        stats = {
            'num_nodes': data.num_nodes if hasattr(data, 'num_nodes') else data.x.size(0),
            'num_edges': data.edge_index.size(1) if hasattr(data, 'edge_index') else 0,
            'num_features': data.x.size(1) if hasattr(data, 'x') else 0,
            'file_size_mb': os.path.getsize(graph_path) / (1024 * 1024),
        }

        # Additional edge types if available
        if hasattr(data, 'triple_index'):
            stats['num_triples'] = data.triple_index.size(1)
        else:
            stats['num_triples'] = 0

        if hasattr(data, 'quadra_index'):
            stats['num_quadras'] = data.quadra_index.size(1)
        else:
            stats['num_quadras'] = 0

        # Compute approximate memory requirement (rough estimate)
        # Formula: nodes * features * 4 bytes (float32) + edges * 8 bytes (int64)
        approx_memory_mb = (
            stats['num_nodes'] * stats['num_features'] * 4 / (1024 * 1024) +
            stats['num_edges'] * 8 / (1024 * 1024) +
            stats['num_triples'] * 12 / (1024 * 1024) +
            stats['num_quadras'] * 16 / (1024 * 1024)
        )
        stats['approx_memory_mb'] = approx_memory_mb

        return stats, None

    except Exception as e:
        return None, str(e)


def analyze_all_graphs(graph_dir, splits_file=None):
    """
    Analyze all graphs in the directory.

    Args:
        graph_dir: Directory containing graph .pt files
        splits_file: Optional splits file to filter samples

    Returns:
        DataFrame with statistics for all graphs
    """
    graph_dir = Path(graph_dir)

    # Get list of samples to analyze
    if splits_file and Path(splits_file).exists():
        print(f"Loading samples from splits file: {splits_file}")
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        sample_ids = splits['train'] + splits['val'] + splits.get('test', [])
        graph_files = [graph_dir / f"{sid}.pt" for sid in sample_ids]
        graph_files = [f for f in graph_files if f.exists()]
    else:
        print(f"Scanning all .pt files in {graph_dir}")
        graph_files = list(graph_dir.glob("*.pt"))

    print(f"Found {len(graph_files)} graph files to analyze")

    # Analyze each graph
    results = []
    errors = []

    for graph_path in tqdm(graph_files, desc="Analyzing graphs"):
        sample_id = graph_path.stem
        stats, error = get_graph_statistics(graph_path)

        if stats is not None:
            stats['sample_id'] = sample_id
            stats['file_path'] = str(graph_path)
            results.append(stats)
        else:
            errors.append({'sample_id': sample_id, 'error': error})

    # Create DataFrame
    df = pd.DataFrame(results)

    if errors:
        print(f"\n⚠️  Failed to analyze {len(errors)} samples:")
        for err in errors[:5]:
            print(f"  - {err['sample_id']}: {err['error']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    return df, errors


def print_statistics_summary(df):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("GRAPH SIZE STATISTICS")
    print("="*70)

    print(f"\nTotal samples analyzed: {len(df)}")

    print("\n--- Number of Nodes ---")
    print(f"  Mean:   {df['num_nodes'].mean():.1f}")
    print(f"  Median: {df['num_nodes'].median():.1f}")
    print(f"  Std:    {df['num_nodes'].std():.1f}")
    print(f"  Min:    {df['num_nodes'].min()}")
    print(f"  Max:    {df['num_nodes'].max()}")
    print(f"  95th:   {df['num_nodes'].quantile(0.95):.1f}")
    print(f"  99th:   {df['num_nodes'].quantile(0.99):.1f}")

    print("\n--- Number of Edges ---")
    print(f"  Mean:   {df['num_edges'].mean():.1f}")
    print(f"  Median: {df['num_edges'].median():.1f}")
    print(f"  Std:    {df['num_edges'].std():.1f}")
    print(f"  Min:    {df['num_edges'].min()}")
    print(f"  Max:    {df['num_edges'].max()}")
    print(f"  95th:   {df['num_edges'].quantile(0.95):.1f}")
    print(f"  99th:   {df['num_edges'].quantile(0.99):.1f}")

    print("\n--- Approximate Memory (MB) ---")
    print(f"  Mean:   {df['approx_memory_mb'].mean():.1f}")
    print(f"  Median: {df['approx_memory_mb'].median():.1f}")
    print(f"  Std:    {df['approx_memory_mb'].std():.1f}")
    print(f"  Min:    {df['approx_memory_mb'].min():.1f}")
    print(f"  Max:    {df['approx_memory_mb'].max():.1f}")
    print(f"  95th:   {df['approx_memory_mb'].quantile(0.95):.1f}")
    print(f"  99th:   {df['approx_memory_mb'].quantile(0.99):.1f}")

    print("\n--- File Size (MB) ---")
    print(f"  Mean:   {df['file_size_mb'].mean():.2f}")
    print(f"  Median: {df['file_size_mb'].median():.2f}")
    print(f"  Total:  {df['file_size_mb'].sum():.2f} MB")

    print("="*70)


def filter_samples(df, max_nodes=None, max_edges=None, max_memory_mb=None,
                  max_file_size_mb=None, percentile_threshold=None):
    """
    Filter samples based on size thresholds.

    Args:
        df: DataFrame with graph statistics
        max_nodes: Maximum number of nodes
        max_edges: Maximum number of edges
        max_memory_mb: Maximum approximate memory in MB
        max_file_size_mb: Maximum file size in MB
        percentile_threshold: Remove top X percentile (e.g., 99 removes top 1%)

    Returns:
        Filtered DataFrame and removed samples
    """
    original_count = len(df)
    removed = pd.DataFrame()

    # Apply percentile threshold first
    if percentile_threshold is not None:
        threshold_nodes = df['num_nodes'].quantile(percentile_threshold / 100)
        threshold_edges = df['num_edges'].quantile(percentile_threshold / 100)

        print(f"\n📊 Percentile-based filtering (removing top {100-percentile_threshold}%):")
        print(f"  Node threshold: {threshold_nodes:.0f}")
        print(f"  Edge threshold: {threshold_edges:.0f}")

        mask = (df['num_nodes'] <= threshold_nodes) | (df['num_edges'] <= threshold_edges)
        removed = pd.concat([removed, df[~mask]])
        df = df[mask]
        print(f"  Removed: {original_count - len(df)} samples")

    # Apply absolute thresholds
    if max_nodes is not None:
        print(f"\n🔢 Filtering by max nodes: {max_nodes}")
        mask = df['num_nodes'] <= max_nodes
        removed = pd.concat([removed, df[~mask]])
        df = df[mask]
        print(f"  Removed: {original_count - len(df)} samples")

    if max_edges is not None:
        print(f"\n🔗 Filtering by max edges: {max_edges}")
        mask = df['num_edges'] <= max_edges
        removed = pd.concat([removed, df[~mask]])
        df = df[mask]
        print(f"  Removed: {original_count - len(df)} samples")

    if max_memory_mb is not None:
        print(f"\n💾 Filtering by max memory: {max_memory_mb} MB")
        mask = df['approx_memory_mb'] <= max_memory_mb
        removed = pd.concat([removed, df[~mask]])
        df = df[mask]
        print(f"  Removed: {original_count - len(df)} samples")

    if max_file_size_mb is not None:
        print(f"\n📁 Filtering by max file size: {max_file_size_mb} MB")
        mask = df['file_size_mb'] <= max_file_size_mb
        removed = pd.concat([removed, df[~mask]])
        df = df[mask]
        print(f"  Removed: {original_count - len(df)} samples")

    # Remove duplicates in removed samples
    removed = removed.drop_duplicates(subset=['sample_id'])

    print(f"\n✅ Final result:")
    print(f"  Original samples: {original_count}")
    print(f"  Kept samples: {len(df)}")
    print(f"  Removed samples: {len(removed)}")
    print(f"  Retention rate: {len(df)/original_count*100:.1f}%")

    return df, removed


def create_filtered_splits(original_splits_file, filtered_sample_ids, output_file):
    """
    Create new splits file with filtered samples.

    Args:
        original_splits_file: Path to original splits.json
        filtered_sample_ids: List of sample IDs to keep
        output_file: Path to save filtered splits
    """
    if not Path(original_splits_file).exists():
        print(f"⚠️  Original splits file not found: {original_splits_file}")
        print("Creating new splits from filtered samples...")

        # Create new splits
        sample_ids = list(filtered_sample_ids)
        np.random.shuffle(sample_ids)

        n_train = int(len(sample_ids) * 0.8)
        n_val = int(len(sample_ids) * 0.1)

        splits = {
            'train': sample_ids[:n_train],
            'val': sample_ids[n_train:n_train+n_val],
            'test': sample_ids[n_train+n_val:]
        }
    else:
        print(f"Loading original splits from {original_splits_file}")
        with open(original_splits_file, 'r') as f:
            original_splits = json.load(f)

        # Filter each split
        filtered_ids_set = set(filtered_sample_ids)
        splits = {
            'train': [sid for sid in original_splits['train'] if sid in filtered_ids_set],
            'val': [sid for sid in original_splits['val'] if sid in filtered_ids_set],
            'test': [sid for sid in original_splits.get('test', []) if sid in filtered_ids_set]
        }

    # Save filtered splits
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"\n💾 Saved filtered splits to: {output_file}")
    print(f"  Train: {len(splits['train'])} samples")
    print(f"  Val: {len(splits['val'])} samples")
    print(f"  Test: {len(splits['test'])} samples")

    return splits


def plot_distribution(df, removed_df, output_dir):
    """
    Plot distribution of graph sizes.

    Args:
        df: DataFrame with kept samples
        removed_df: DataFrame with removed samples
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Number of nodes
    ax = axes[0, 0]
    ax.hist(df['num_nodes'], bins=50, alpha=0.7, label='Kept', color='green', edgecolor='black')
    if len(removed_df) > 0:
        ax.hist(removed_df['num_nodes'], bins=50, alpha=0.7, label='Removed', color='red', edgecolor='black')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Number of Nodes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Number of edges
    ax = axes[0, 1]
    ax.hist(df['num_edges'], bins=50, alpha=0.7, label='Kept', color='green', edgecolor='black')
    if len(removed_df) > 0:
        ax.hist(removed_df['num_edges'], bins=50, alpha=0.7, label='Removed', color='red', edgecolor='black')
    ax.set_xlabel('Number of Edges')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Number of Edges')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Approximate memory
    ax = axes[1, 0]
    ax.hist(df['approx_memory_mb'], bins=50, alpha=0.7, label='Kept', color='green', edgecolor='black')
    if len(removed_df) > 0:
        ax.hist(removed_df['approx_memory_mb'], bins=50, alpha=0.7, label='Removed', color='red', edgecolor='black')
    ax.set_xlabel('Approximate Memory (MB)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Approximate Memory Usage')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: File size
    ax = axes[1, 1]
    ax.hist(df['file_size_mb'], bins=50, alpha=0.7, label='Kept', color='green', edgecolor='black')
    if len(removed_df) > 0:
        ax.hist(removed_df['file_size_mb'], bins=50, alpha=0.7, label='Removed', color='red', edgecolor='black')
    ax.set_xlabel('File Size (MB)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of File Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "graph_size_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved distribution plot to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Filter out overly large graph samples to prevent OOM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - auto filter top 5% largest samples
  python scripts/filter_large_samples.py

  # Filter with specific thresholds
  python scripts/filter_large_samples.py --max_nodes 1000 --max_edges 5000

  # Filter top 1% (99th percentile)
  python scripts/filter_large_samples.py --percentile 99

  # Custom output location
  python scripts/filter_large_samples.py --output_splits data/splits/filtered_splits.json
        """
    )

    parser.add_argument("--graph_dir", type=str, default="data/processed/graphs",
                        help="Directory containing graph .pt files")
    parser.add_argument("--splits_file", type=str, default="data/splits/splits.json",
                        help="Original splits file (optional)")
    parser.add_argument("--output_splits", type=str, default="data/splits/filtered_splits.json",
                        help="Output path for filtered splits")
    parser.add_argument("--output_dir", type=str, default="data/analysis",
                        help="Directory for output files (stats, plots)")

    # Filtering options
    parser.add_argument("--max_nodes", type=int, default=None,
                        help="Maximum number of nodes (absolute threshold)")
    parser.add_argument("--max_edges", type=int, default=None,
                        help="Maximum number of edges (absolute threshold)")
    parser.add_argument("--max_memory_mb", type=float, default=None,
                        help="Maximum approximate memory in MB")
    parser.add_argument("--max_file_size_mb", type=float, default=None,
                        help="Maximum file size in MB")
    parser.add_argument("--percentile", type=float, default=95,
                        help="Keep samples below this percentile (default: 95, removes top 5%%)")

    parser.add_argument("--no_plot", action="store_true",
                        help="Skip generating plots")
    parser.add_argument("--no_filter", action="store_true",
                        help="Only analyze, don't filter")

    args = parser.parse_args()

    print("="*70)
    print("GRAPH SAMPLE PRE-FILTERING TOOL")
    print("="*70)

    # Step 1: Analyze all graphs
    print("\n[Step 1/4] Analyzing all graph samples...")
    df, errors = analyze_all_graphs(args.graph_dir, args.splits_file)

    if len(df) == 0:
        print("❌ No valid samples found!")
        return

    # Step 2: Print statistics
    print("\n[Step 2/4] Computing statistics...")
    print_statistics_summary(df)

    # Save statistics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_file = output_dir / "graph_statistics.csv"
    df.to_csv(stats_file, index=False)
    print(f"\n💾 Saved detailed statistics to: {stats_file}")

    # Step 3: Filter samples
    if not args.no_filter:
        print("\n[Step 3/4] Filtering samples...")
        filtered_df, removed_df = filter_samples(
            df,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            max_memory_mb=args.max_memory_mb,
            max_file_size_mb=args.max_file_size_mb,
            percentile_threshold=args.percentile
        )

        # Save removed samples list
        if len(removed_df) > 0:
            removed_file = output_dir / "removed_samples.csv"
            removed_df.to_csv(removed_file, index=False)
            print(f"\n💾 Saved removed samples list to: {removed_file}")

            print("\n🔴 Top 10 largest removed samples:")
            top_removed = removed_df.nlargest(10, 'num_nodes')[['sample_id', 'num_nodes', 'num_edges', 'approx_memory_mb']]
            print(top_removed.to_string(index=False))

        # Create filtered splits
        print("\n[Step 4/4] Creating filtered dataset splits...")
        filtered_ids = filtered_df['sample_id'].tolist()
        splits = create_filtered_splits(
            args.splits_file,
            filtered_ids,
            args.output_splits
        )

        # Plot distributions
        if not args.no_plot:
            print("\nGenerating distribution plots...")
            plot_distribution(filtered_df, removed_df, output_dir)
    else:
        print("\n[Step 3/4] Skipping filtering (--no_filter specified)")
        print("[Step 4/4] Skipping split creation")

    print("\n" + "="*70)
    print("✅ FILTERING COMPLETE!")
    print("="*70)
    print(f"\nTo use the filtered dataset, run training with:")
    print(f"  --splits_file {args.output_splits}")
    print("\nExample:")
    print(f"  python scripts/04_train_model.py \\")
    print(f"      --splits_file {args.output_splits} \\")
    print(f"      --use_amp --use_ddp --world_size 4")
    print("="*70)


if __name__ == "__main__":
    main()
