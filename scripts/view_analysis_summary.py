#!/usr/bin/env python3
"""
Quick Summary Viewer for Embedding Analysis Results

This script provides a quick overview of analysis results without
needing to open multiple files.

Usage:
    python scripts/view_analysis_summary.py --results_dir results/embedding_analysis
"""

import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np

def print_header(text, width=70, char='='):
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}\n")

def print_section(text, width=70, char='-'):
    """Print a section separator."""
    print(f"\n{char * width}")
    print(f"{text}")
    print(f"{char * width}\n")

def load_and_summarize_distances(results_dir):
    """Load and summarize distance metrics."""
    print_section("Distance Metrics Summary")

    dist_file = results_dir / 'visualizations' / 'pocket_ligand_distances.csv'

    if not dist_file.exists():
        print(f"⚠️  File not found: {dist_file}")
        return

    df = pd.read_csv(dist_file)

    print(f"Total pocket-ligand pairs: {len(df)}")
    print(f"\nCosine Distance Statistics:")
    print(f"  Mean:   {df['cosine_distance'].mean():.4f}")
    print(f"  Median: {df['cosine_distance'].median():.4f}")
    print(f"  Std:    {df['cosine_distance'].std():.4f}")
    print(f"  Min:    {df['cosine_distance'].min():.4f}")
    print(f"  Max:    {df['cosine_distance'].max():.4f}")

    print(f"\nCosine Similarity Statistics:")
    print(f"  Mean:   {df['cosine_similarity'].mean():.4f}")
    print(f"  Median: {df['cosine_similarity'].median():.4f}")
    print(f"  Std:    {df['cosine_similarity'].std():.4f}")

    print(f"\nEuclidean Distance Statistics:")
    print(f"  Mean:   {df['euclidean_distance'].mean():.4f}")
    print(f"  Median: {df['euclidean_distance'].median():.4f}")
    print(f"  Std:    {df['euclidean_distance'].std():.4f}")

    # Top 10 closest pairs
    print(f"\n🎯 Top 10 Closest Pocket-Ligand Pairs (by cosine distance):")
    top10 = df.nsmallest(10, 'cosine_distance')
    for i, row in top10.iterrows():
        print(f"  {row['pocket_id']:30s} - {row['ligand_name']:10s}  Distance: {row['cosine_distance']:.4f}")

    # Top 10 furthest pairs
    print(f"\n⚠️  Top 10 Furthest Pocket-Ligand Pairs (by cosine distance):")
    bottom10 = df.nlargest(10, 'cosine_distance')
    for i, row in bottom10.iterrows():
        print(f"  {row['pocket_id']:30s} - {row['ligand_name']:10s}  Distance: {row['cosine_distance']:.4f}")


def load_and_summarize_ligands(results_dir):
    """Load and summarize ligand statistics."""
    print_section("Ligand Summary")

    summary_file = results_dir / 'visualizations' / 'ligand_summary.csv'

    if not summary_file.exists():
        print(f"⚠️  File not found: {summary_file}")
        return

    df = pd.read_csv(summary_file)

    print(f"Total unique ligands: {len(df)}")
    print(f"Total pockets: {df['n_pockets'].sum()}")
    print(f"\nLigand distribution:")
    print(f"  Mean pockets per ligand: {df['n_pockets'].mean():.1f}")
    print(f"  Median pockets per ligand: {df['n_pockets'].median():.1f}")
    print(f"  Max pockets for one ligand: {df['n_pockets'].max()}")

    print(f"\n🏆 Top 20 Most Common Ligands:")
    top20 = df.head(20)
    for i, row in top20.iterrows():
        bar_length = int(row['n_pockets'] / df['n_pockets'].max() * 40)
        bar = '█' * bar_length
        print(f"  {row['ligand_name']:15s} {row['n_pockets']:4d} {bar}")


def load_and_summarize_retrieval(results_dir):
    """Load and summarize retrieval performance."""
    print_section("Retrieval Performance")

    retrieval_file = results_dir / 'advanced_analysis' / 'retrieval_results.csv'

    if not retrieval_file.exists():
        print(f"⚠️  File not found: {retrieval_file}")
        return

    df = pd.read_csv(retrieval_file)

    top1_acc = (df['rank'] == 1).mean()
    top5_acc = (df['rank'] <= 5).mean()
    top10_acc = (df['rank'] <= 10).mean()
    top20_acc = (df['rank'] <= 20).mean()
    mrr = (1.0 / df['rank']).mean()
    mean_rank = df['rank'].mean()
    median_rank = df['rank'].median()

    print(f"Total queries: {len(df)}")
    print(f"\n📊 Accuracy Metrics:")
    print(f"  Top-1:   {top1_acc:.4f} ({top1_acc*100:.1f}%)")
    print(f"  Top-5:   {top5_acc:.4f} ({top5_acc*100:.1f}%)")
    print(f"  Top-10:  {top10_acc:.4f} ({top10_acc*100:.1f}%)")
    print(f"  Top-20:  {top20_acc:.4f} ({top20_acc*100:.1f}%)")

    print(f"\n📈 Ranking Metrics:")
    print(f"  Mean Rank:       {mean_rank:.2f}")
    print(f"  Median Rank:     {median_rank:.1f}")
    print(f"  MRR (Mean Reciprocal Rank): {mrr:.4f}")

    # Rank distribution
    print(f"\n📉 Rank Distribution:")
    rank_bins = [1, 2, 5, 10, 20, 50, 100, float('inf')]
    rank_labels = ['1', '2-5', '6-10', '11-20', '21-50', '51-100', '>100']

    for i in range(len(rank_bins) - 1):
        lower = rank_bins[i]
        upper = rank_bins[i + 1]
        if upper == float('inf'):
            count = (df['rank'] >= lower).sum()
        else:
            count = ((df['rank'] >= lower) & (df['rank'] < upper)).sum()
        pct = count / len(df) * 100
        bar_length = int(pct / 100 * 40)
        bar = '█' * bar_length
        print(f"  Rank {rank_labels[i]:>6s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Retrieval failures
    failures = df[df['rank'] > 1]
    if len(failures) > 0:
        print(f"\n⚠️  Retrieval Failures (rank > 1): {len(failures)} ({len(failures)/len(df)*100:.1f}%)")
        print(f"Sample failures:")
        for i, row in failures.head(10).iterrows():
            print(f"  {row['pocket_id']:30s} True: {row['true_ligand']:10s} Rank: {row['rank']:3d}  Predicted: {row['top1']:10s}")


def load_and_summarize_clustering(results_dir):
    """Load and summarize clustering results."""
    print_section("Clustering Analysis")

    cluster_file = results_dir / 'advanced_analysis' / 'cluster_assignments.csv'

    if not cluster_file.exists():
        print(f"⚠️  File not found: {cluster_file}")
        return

    df = pd.read_csv(cluster_file)

    n_clusters = df['cluster'].nunique()

    print(f"Number of clusters: {n_clusters}")
    print(f"Total pockets: {len(df)}")

    print(f"\n🔍 Cluster Composition:")
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        n_pockets = len(cluster_data)
        ligand_counts = cluster_data['ligand_name'].value_counts()
        top_ligands = ligand_counts.head(5)

        print(f"\n  Cluster {cluster_id} (n={n_pockets}):")
        for ligand, count in top_ligands.items():
            pct = count / n_pockets * 100
            print(f"    {ligand:15s}: {count:4d} ({pct:5.1f}%)")


def load_and_summarize_intra_inter(results_dir):
    """Load and summarize intra/inter-ligand distances."""
    print_section("Intra-Ligand vs Inter-Ligand Distances")

    dist_file = results_dir / 'advanced_analysis' / 'intra_inter_distances.csv'

    if not dist_file.exists():
        print(f"⚠️  File not found: {dist_file}")
        return

    df = pd.read_csv(dist_file)

    intra = df[df['type'] == 'intra']
    inter = df[df['type'] == 'inter']

    if len(intra) > 0 and len(inter) > 0:
        print(f"Intra-ligand distances (same ligand): {len(intra)}")
        print(f"  Mean:   {intra['distance'].mean():.4f}")
        print(f"  Median: {intra['distance'].median():.4f}")
        print(f"  Std:    {intra['distance'].std():.4f}")

        print(f"\nInter-ligand distances (different ligands): {len(inter)}")
        print(f"  Mean:   {inter['distance'].mean():.4f}")
        print(f"  Median: {inter['distance'].median():.4f}")
        print(f"  Std:    {inter['distance'].std():.4f}")

        # Separation quality
        intra_mean = intra['distance'].mean()
        inter_mean = inter['distance'].mean()
        separation = (inter_mean - intra_mean) / intra['distance'].std()

        print(f"\n📊 Separation Quality:")
        print(f"  Difference (inter - intra): {inter_mean - intra_mean:.4f}")
        print(f"  Separation score (z-score): {separation:.2f}")

        if separation > 2:
            print(f"  ✅ Excellent separation (score > 2)")
        elif separation > 1:
            print(f"  ✓ Good separation (score > 1)")
        elif separation > 0.5:
            print(f"  ⚠️  Moderate separation (score > 0.5)")
        else:
            print(f"  ❌ Poor separation (score < 0.5)")
    else:
        print("⚠️  Insufficient data for analysis")


def view_files(results_dir):
    """List all output files."""
    print_section("Output Files")

    viz_dir = results_dir / 'visualizations'
    adv_dir = results_dir / 'advanced_analysis'

    print("📁 Visualization Files:")
    if viz_dir.exists():
        files = sorted(viz_dir.iterdir())
        for f in files:
            size = f.stat().st_size / 1024  # KB
            print(f"  {f.name:50s} ({size:>8.1f} KB)")
    else:
        print("  ⚠️  Directory not found")

    print("\n📁 Advanced Analysis Files:")
    if adv_dir.exists():
        files = sorted(adv_dir.iterdir())
        for f in files:
            size = f.stat().st_size / 1024  # KB
            print(f"  {f.name:50s} ({size:>8.1f} KB)")
    else:
        print("  ⚠️  Directory not found")


def main():
    parser = argparse.ArgumentParser(
        description="View summary of embedding analysis results"
    )
    parser.add_argument("--results_dir", type=str, default="results/embedding_analysis",
                       help="Results directory (default: results/embedding_analysis)")
    parser.add_argument("--sections", nargs='+',
                       choices=['distances', 'ligands', 'retrieval', 'clustering', 'intra_inter', 'files', 'all'],
                       default=['all'],
                       help="Sections to display (default: all)")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print_header("Embedding Analysis Results Summary")

    print(f"Results directory: {results_dir}")
    print(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    sections = args.sections
    if 'all' in sections:
        sections = ['distances', 'ligands', 'retrieval', 'clustering', 'intra_inter', 'files']

    # Display requested sections
    if 'distances' in sections:
        load_and_summarize_distances(results_dir)

    if 'ligands' in sections:
        load_and_summarize_ligands(results_dir)

    if 'retrieval' in sections:
        load_and_summarize_retrieval(results_dir)

    if 'clustering' in sections:
        load_and_summarize_clustering(results_dir)

    if 'intra_inter' in sections:
        load_and_summarize_intra_inter(results_dir)

    if 'files' in sections:
        view_files(results_dir)

    print("\n" + "="*70)
    print("For detailed reports, see:")
    print(f"  - {results_dir / 'visualizations' / 'analysis_report.md'}")
    print(f"  - Individual CSV files in {results_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
