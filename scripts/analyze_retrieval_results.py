#!/usr/bin/env python3
"""
Retrieval Results Analysis and Visualization

This script analyzes the output from evaluate_retrieval.py and generates:
- Statistical summaries
- Distribution plots
- Per-ligand performance analysis
- Error analysis
"""
import argparse
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_path):
    """Load results JSON from evaluate_retrieval.py."""
    print(f"Loading results from {results_path}...")

    with open(results_path, 'r') as f:
        results = json.load(f)

    print(f"✓ Loaded results with {results['successful_predictions']} successful predictions")
    return results


def analyze_per_ligand_performance(results):
    """
    Analyze performance breakdown by ligand type.

    Returns:
        Dictionary with per-ligand statistics
    """
    print("\n" + "="*70)
    print("Per-Ligand Performance Analysis")
    print("="*70)

    ligand_stats = defaultdict(lambda: {
        'ranks': [],
        'normalized_ranks': [],
        'reciprocal_ranks': [],
        'count': 0
    })

    for sample in results['detailed_results']:
        ligand_id = sample['true_ligand_id']
        ligand_stats[ligand_id]['ranks'].append(sample['rank'])
        ligand_stats[ligand_id]['normalized_ranks'].append(sample['normalized_rank'])
        ligand_stats[ligand_id]['reciprocal_ranks'].append(sample['reciprocal_rank'])
        ligand_stats[ligand_id]['count'] += 1

    # Compute summary statistics for each ligand
    ligand_summary = {}
    for ligand_id, stats in ligand_stats.items():
        ligand_summary[ligand_id] = {
            'count': stats['count'],
            'mean_rank': float(np.mean(stats['ranks'])),
            'mean_normalized_rank': float(np.mean(stats['normalized_ranks'])),
            'mean_reciprocal_rank': float(np.mean(stats['reciprocal_ranks'])),
            'top1_accuracy': float(sum(1 for r in stats['ranks'] if r == 1) / len(stats['ranks']) * 100)
        }

    # Sort by mean normalized rank
    sorted_ligands = sorted(ligand_summary.items(),
                           key=lambda x: x[1]['mean_normalized_rank'])

    print(f"\nFound {len(sorted_ligands)} unique ligands\n")
    print(f"{'Ligand':<15} {'Count':<8} {'Mean Rank':<12} {'Norm Rank':<12} {'MRR':<10} {'Top-1 Acc':<10}")
    print("-" * 70)

    for ligand_id, stats in sorted_ligands[:20]:  # Top 20
        print(f"{ligand_id:<15} {stats['count']:<8} "
              f"{stats['mean_rank']:<12.2f} "
              f"{stats['mean_normalized_rank']:<12.4f} "
              f"{stats['mean_reciprocal_rank']:<10.4f} "
              f"{stats['top1_accuracy']:<10.1f}%")

    if len(sorted_ligands) > 20:
        print(f"\n... and {len(sorted_ligands) - 20} more ligands")

    return ligand_summary


def analyze_error_cases(results, top_n=20):
    """
    Analyze worst-performing cases.

    Args:
        results: Results dictionary
        top_n: Number of worst cases to show
    """
    print("\n" + "="*70)
    print(f"Top {top_n} Worst-Performing Cases")
    print("="*70)

    # Sort by normalized rank (descending)
    sorted_samples = sorted(results['detailed_results'],
                           key=lambda x: x['normalized_rank'],
                           reverse=True)

    print(f"\n{'Complex ID':<30} {'True Ligand':<12} {'Rank':<8} {'Norm Rank':<12} {'Top-1 Pred':<12}")
    print("-" * 70)

    for sample in sorted_samples[:top_n]:
        top1_pred = sample['top_5_predictions'][0]['ligand_id'] if sample['top_5_predictions'] else 'N/A'
        print(f"{sample['complex_id']:<30} "
              f"{sample['true_ligand_id']:<12} "
              f"{sample['rank']:<8} "
              f"{sample['normalized_rank']:<12.4f} "
              f"{top1_pred:<12}")

    return sorted_samples[:top_n]


def plot_rank_distribution(results, output_dir):
    """Plot distribution of ranks."""
    print("\nGenerating rank distribution plot...")

    ranks = [sample['rank'] for sample in results['detailed_results']]
    normalized_ranks = [sample['normalized_rank'] for sample in results['detailed_results']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Absolute ranks
    axes[0].hist(ranks, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(ranks), color='red', linestyle='--',
                   label=f'Mean: {np.mean(ranks):.1f}')
    axes[0].axvline(np.median(ranks), color='green', linestyle='--',
                   label=f'Median: {np.median(ranks):.1f}')
    axes[0].set_xlabel('Rank', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Absolute Ranks', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot 2: Normalized ranks
    axes[1].hist(normalized_ranks, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(np.mean(normalized_ranks), color='red', linestyle='--',
                   label=f'Mean: {np.mean(normalized_ranks):.3f}')
    axes[1].axvline(np.median(normalized_ranks), color='green', linestyle='--',
                   label=f'Median: {np.median(normalized_ranks):.3f}')
    axes[1].set_xlabel('Normalized Rank (0-1)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Normalized Ranks', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir) / 'rank_distribution.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")

    plt.close()


def plot_recall_curve(results, output_dir):
    """Plot Recall@K curve."""
    print("\nGenerating Recall@K curve...")

    ranks = np.array([sample['rank'] for sample in results['detailed_results']])
    total_samples = len(ranks)

    # Calculate recall at different K values
    k_values = list(range(1, min(101, results['total_baits'] + 1)))
    recall_values = []

    for k in k_values:
        recall = np.sum(ranks <= k) / total_samples * 100
        recall_values.append(recall)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, recall_values, linewidth=2, marker='o', markersize=3)

    # Add reference lines
    for k_ref in [1, 5, 10, 20, 50]:
        if k_ref <= max(k_values):
            recall_at_k = recall_values[k_ref - 1]
            plt.axvline(k_ref, color='gray', linestyle='--', alpha=0.5)
            plt.text(k_ref, recall_at_k + 2, f'K={k_ref}\n{recall_at_k:.1f}%',
                    ha='center', fontsize=9)

    plt.xlabel('K (Top-K Predictions)', fontsize=12)
    plt.ylabel('Recall@K (%)', fontsize=12)
    plt.title('Recall@K Curve', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.xlim(0, min(100, max(k_values)))
    plt.ylim(0, 105)

    output_path = Path(output_dir) / 'recall_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")

    plt.close()


def plot_cumulative_distribution(results, output_dir):
    """Plot cumulative distribution of normalized ranks."""
    print("\nGenerating cumulative distribution plot...")

    normalized_ranks = np.array([sample['normalized_rank']
                                 for sample in results['detailed_results']])

    # Sort for cumulative plot
    sorted_ranks = np.sort(normalized_ranks)
    cumulative = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks) * 100

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_ranks, cumulative, linewidth=2)

    # Add reference lines
    for percentile in [25, 50, 75, 90]:
        rank_at_percentile = np.percentile(normalized_ranks, percentile)
        plt.axhline(percentile, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(rank_at_percentile, color='gray', linestyle='--', alpha=0.5)
        plt.text(rank_at_percentile + 0.02, percentile + 2,
                f'{percentile}th: {rank_at_percentile:.3f}', fontsize=9)

    plt.xlabel('Normalized Rank', fontsize=12)
    plt.ylabel('Cumulative Percentage (%)', fontsize=12)
    plt.title('Cumulative Distribution of Normalized Ranks',
             fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 105)

    output_path = Path(output_dir) / 'cumulative_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")

    plt.close()


def generate_summary_report(results, ligand_summary, output_path):
    """Generate a text summary report."""
    print(f"\nGenerating summary report...")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("RETRIEVAL EVALUATION SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")

        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total samples: {results['total_samples']}\n")
        f.write(f"Successful predictions: {results['successful_predictions']}\n")
        f.write(f"Failed predictions: {len(results['failed_predictions'])}\n")
        f.write(f"Total baits in library: {results['total_baits']}\n\n")

        # Key metrics
        if 'metrics' in results:
            metrics = results['metrics']

            f.write("\nKey Metrics:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Mean Normalized Rank: {metrics['mean_normalized_rank']:.4f}\n")
            f.write(f"Median Normalized Rank: {metrics['median_normalized_rank']:.4f}\n")
            f.write(f"Mean Reciprocal Rank (MRR): {metrics['mrr']:.4f}\n")
            f.write(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%\n")
            f.write(f"\nMean Rank: {metrics['mean_rank']:.2f}\n")
            f.write(f"Median Rank: {metrics['median_rank']:.1f}\n\n")

            # Recall@K
            f.write("\nRecall@K:\n")
            f.write("-" * 70 + "\n")
            for key in sorted(metrics['recall'].keys()):
                data = metrics['recall'][key]
                f.write(f"{key}: {data['percentage']:.2f}% ({data['hits']}/{data['total']})\n")

        # Per-ligand summary
        f.write("\n\nTop 10 Best-Performing Ligands:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Ligand':<15} {'Count':<8} {'Mean Rank':<12} {'Norm Rank':<12} {'Top-1 Acc':<10}\n")
        f.write("-" * 70 + "\n")

        sorted_ligands = sorted(ligand_summary.items(),
                               key=lambda x: x[1]['mean_normalized_rank'])

        for ligand_id, stats in sorted_ligands[:10]:
            f.write(f"{ligand_id:<15} {stats['count']:<8} "
                   f"{stats['mean_rank']:<12.2f} "
                   f"{stats['mean_normalized_rank']:<12.4f} "
                   f"{stats['top1_accuracy']:<10.1f}%\n")

        f.write("\n\nTop 10 Worst-Performing Ligands:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Ligand':<15} {'Count':<8} {'Mean Rank':<12} {'Norm Rank':<12} {'Top-1 Acc':<10}\n")
        f.write("-" * 70 + "\n")

        for ligand_id, stats in sorted_ligands[-10:]:
            f.write(f"{ligand_id:<15} {stats['count']:<8} "
                   f"{stats['mean_rank']:<12.2f} "
                   f"{stats['mean_normalized_rank']:<12.4f} "
                   f"{stats['top1_accuracy']:<10.1f}%\n")

    print(f"✓ Saved to {output_path}")


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize retrieval evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--results", type=str, required=True,
                        help="Path to results JSON from evaluate_retrieval.py")
    parser.add_argument("--output_dir", type=str, default="results/retrieval_analysis",
                        help="Output directory for plots and reports")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip generating plots (text analysis only)")

    args = parser.parse_args()

    # Load results
    results = load_results(args.results)

    # Per-ligand analysis
    ligand_summary = analyze_per_ligand_performance(results)

    # Error analysis
    worst_cases = analyze_error_cases(results, top_n=20)

    # Generate plots
    if not args.no_plots:
        try:
            plot_rank_distribution(results, args.output_dir)
            plot_recall_curve(results, args.output_dir)
            plot_cumulative_distribution(results, args.output_dir)
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to generate some plots: {e}")
            print("Continuing with text analysis...")

    # Generate summary report
    report_path = Path(args.output_dir) / "summary_report.txt"
    generate_summary_report(results, ligand_summary, report_path)

    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
