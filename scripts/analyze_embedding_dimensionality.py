#!/usr/bin/env python3
"""
Analyze the intrinsic dimensionality of ligand embeddings.

This script helps answer: Do we really need 1536 dimensions?
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path


def analyze_embeddings(embeddings_path):
    """
    Analyze ligand embedding dimensionality.

    Args:
        embeddings_path: Path to HDF5 file with embeddings
    """
    print("="*80)
    print("Ligand Embedding Dimensionality Analysis")
    print("="*80)

    # Load embeddings
    print(f"\nLoading embeddings from: {embeddings_path}")
    embeddings = []
    keys = []

    with h5py.File(embeddings_path, 'r') as f:
        for key in f.keys():
            embeddings.append(f[key][:])
            keys.append(key)

    embeddings = np.array(embeddings)
    print(f"Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # Basic statistics
    print(f"\n--- Basic Statistics ---")
    print(f"Mean: {embeddings.mean():.4f}")
    print(f"Std: {embeddings.std():.4f}")
    print(f"Min: {embeddings.min():.4f}")
    print(f"Max: {embeddings.max():.4f}")

    # Check for zeros/sparsity
    zero_ratio = (embeddings == 0).mean()
    print(f"Zero ratio: {zero_ratio:.2%}")

    # L2 norms
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n--- L2 Norms ---")
    print(f"Mean norm: {norms.mean():.4f}")
    print(f"Std norm: {norms.std():.4f}")
    print(f"Min norm: {norms.min():.4f}")
    print(f"Max norm: {norms.max():.4f}")

    # PCA Analysis
    print(f"\n--- PCA Analysis ---")
    print("Computing PCA...")
    pca = PCA()
    pca.fit(embeddings)

    # Explained variance
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)

    # Find dimensions needed for different variance thresholds
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    print("\nDimensions needed to explain variance:")
    for threshold in thresholds:
        n_dims = np.argmax(cumsum_var >= threshold) + 1
        print(f"  {threshold*100:.0f}% variance: {n_dims} dims " +
              f"({n_dims/embeddings.shape[1]*100:.1f}% of original)")

    # Effective rank
    singular_values = pca.singular_values_
    effective_rank = np.sum(singular_values) ** 2 / np.sum(singular_values ** 2)
    print(f"\nEffective rank: {effective_rank:.1f} " +
          f"({effective_rank/embeddings.shape[1]*100:.1f}% of original)")

    # Top eigenvalues
    print(f"\nTop 10 singular values:")
    for i, sv in enumerate(singular_values[:10]):
        var_explained = pca.explained_variance_ratio_[i]
        print(f"  PC{i+1}: {sv:.2f} (explains {var_explained*100:.2f}% variance)")

    # Recommendations
    print(f"\n--- Recommendations ---")
    n_90 = np.argmax(cumsum_var >= 0.9) + 1
    n_95 = np.argmax(cumsum_var >= 0.95) + 1

    if n_90 < 512:
        print(f"✓ Can safely reduce to {n_90} dimensions (preserves 90% variance)")
        print(f"  This is a {(1 - n_90/embeddings.shape[1])*100:.1f}% reduction!")
    elif n_90 < 1024:
        print(f"✓ Can reduce to {n_90} dimensions (preserves 90% variance)")
        print(f"  This is a {(1 - n_90/embeddings.shape[1])*100:.1f}% reduction")
    else:
        print(f"⚠ Embeddings utilize high dimensionality")
        print(f"  {n_90} dims needed for 90% variance")

    print(f"\nFor 95% variance: {n_95} dims needed")

    # Suggest pocket output dimension
    print(f"\n--- Suggested Pocket Output Dimension ---")
    if n_90 < 256:
        suggested = 256
    elif n_90 < 512:
        suggested = 512
    else:
        suggested = min(1024, embeddings.shape[1])

    print(f"Recommended: {suggested} dimensions")
    print(f"Reasoning:")
    print(f"  - Ligand embeddings effectively use ~{n_90} dims")
    print(f"  - Pocket has less structural complexity than full molecule")
    print(f"  - {suggested} dims provides good capacity without overfitting")

    # Save plot
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(12, 5))

    # Subplot 1: Cumulative variance
    plt.subplot(1, 2, 1)
    plt.plot(range(1, min(500, len(cumsum_var))+1),
             cumsum_var[:500],
             linewidth=2)
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% variance')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA: Cumulative Variance Explained')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Individual variance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, min(100, len(pca.explained_variance_ratio_))+1),
             pca.explained_variance_ratio_[:100],
             linewidth=2)
    plt.xlabel('Component Number')
    plt.ylabel('Variance Explained')
    plt.title('PCA: Individual Component Variance')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plot_path = output_dir / "embedding_pca_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {plot_path}")

    # Save detailed results
    results_path = output_dir / "pca_results.txt"
    with open(results_path, 'w') as f:
        f.write("PCA Analysis Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"Original dimension: {embeddings.shape[1]}\n")
        f.write(f"Number of samples: {len(embeddings)}\n\n")

        f.write("Variance explained by dimensions:\n")
        for threshold in thresholds:
            n_dims = np.argmax(cumsum_var >= threshold) + 1
            f.write(f"  {threshold*100:.0f}%: {n_dims} dims\n")

        f.write(f"\nEffective rank: {effective_rank:.1f}\n")
        f.write(f"\nRecommended dimension: {suggested}\n")

    print(f"✓ Results saved to: {results_path}")

    return {
        'n_90': n_90,
        'n_95': n_95,
        'effective_rank': effective_rank,
        'suggested_dim': suggested
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze ligand embedding dimensionality")
    parser.add_argument("--embeddings", type=str,
                        default="data/processed/ligand_embeddings.h5",
                        help="Path to ligand embeddings HDF5 file")

    args = parser.parse_args()

    results = analyze_embeddings(args.embeddings)

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nKey findings:")
    print(f"  - 90% variance preserved with: {results['n_90']} dims")
    print(f"  - Effective rank: {results['effective_rank']:.0f}")
    print(f"  - Recommended pocket output: {results['suggested_dim']} dims")
    print("\nNext steps:")
    print("  1. Consider reducing ligand embeddings to lower dimension")
    print("  2. Train pocket model with recommended output dimension")
    print("  3. Compare performance vs 1536-dim baseline")
    print("="*80)
