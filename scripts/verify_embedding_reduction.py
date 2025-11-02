#!/usr/bin/env python3
"""
Verify the quality of ligand embedding dimensionality reduction.

This script performs comprehensive checks on the reduced embeddings:
1. Normalization status verification
2. Reconstruction error analysis
3. Embedding similarity preservation
4. PCA model validation

Usage:
    python scripts/verify_embedding_reduction.py \
        --original data/processed/ligand_embeddings.h5 \
        --reduced data/processed/ligand_embeddings_256d.h5 \
        --pca_model data/processed/pca_model_256d.pkl \
        --norm_params_256d data/processed/ligand_embedding_norm_params_256d.npz
"""

import argparse
import h5py
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def check_normalization(embeddings_path, embedding_name="", max_samples=None):
    """
    Check if embeddings are properly normalized (z-score normalization).

    For z-score normalization, each dimension should have:
    - Mean ≈ 0 (across all samples)
    - Std ≈ 1 (across all samples)

    Args:
        embeddings_path: Path to HDF5 file
        embedding_name: Name for display
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        dict with normalization statistics
    """
    print(f"\n{'='*80}")
    print(f"Checking Normalization: {embedding_name}")
    print(f"{'='*80}")

    all_embeddings = []
    with h5py.File(embeddings_path, 'r') as f:
        keys = list(f.keys())
        if max_samples:
            keys = keys[:max_samples]
        print(f"Loading {len(keys)}/{len(f.keys())} embeddings...")
        for key in tqdm(keys, desc="Loading"):
            all_embeddings.append(f[key][:])

    all_embeddings = np.array(all_embeddings)
    print(f"Loaded shape: {all_embeddings.shape}")

    # Per-dimension statistics (the key metric for z-score normalization)
    dim_means = all_embeddings.mean(axis=0)  # Mean of each dimension across samples
    dim_stds = all_embeddings.std(axis=0, ddof=0)   # Std of each dimension across samples

    print(f"\n📊 Z-score Normalization Check (per-dimension):")
    print(f"  Expected: Each dimension has mean≈0, std≈1 (across samples)")
    print(f"\n  Dimension Means:")
    print(f"    Average: {dim_means.mean():.6f}")
    print(f"    Std dev: {dim_means.std():.6f}")
    print(f"    Range: [{dim_means.min():.6f}, {dim_means.max():.6f}]")
    print(f"    Dims with |mean| > 0.1: {(np.abs(dim_means) > 0.1).sum()}/{len(dim_means)}")

    print(f"\n  Dimension Stds:")
    print(f"    Average: {dim_stds.mean():.6f}")
    print(f"    Std dev: {dim_stds.std():.6f}")
    print(f"    Range: [{dim_stds.min():.6f}, {dim_stds.max():.6f}]")
    print(f"    Dims with |std-1| > 0.2: {(np.abs(dim_stds - 1.0) > 0.2).sum()}/{len(dim_stds)}")

    # Global statistics (for reference)
    global_mean = all_embeddings.mean()
    global_std = all_embeddings.std()
    print(f"\n  Global Statistics (for reference):")
    print(f"    Mean: {global_mean:.6f}")
    print(f"    Std: {global_std:.6f}")

    # Per-sample statistics
    sample_means = all_embeddings.mean(axis=1)
    sample_stds = all_embeddings.std(axis=1)
    print(f"\n  Per-Sample Statistics:")
    print(f"    Mean range: [{sample_means.min():.4f}, {sample_means.max():.4f}]")
    print(f"    Std range: [{sample_stds.min():.4f}, {sample_stds.max():.4f}]")

    # Normalization verdict based on per-dimension statistics
    mean_check = np.abs(dim_means).max() < 0.15  # Max abs mean < 0.15
    std_check = np.abs(dim_stds - 1.0).max() < 0.3  # Max |std-1| < 0.3
    is_normalized = mean_check and std_check

    if is_normalized:
        print(f"\n✅ Embeddings are properly normalized (z-score)")
        print(f"   All dimensions have |mean| < 0.15 and |std-1| < 0.3")
    else:
        print(f"\n⚠️  Embeddings may NOT be properly normalized")
        if not mean_check:
            print(f"   Issue: Some dimensions have large mean (max |mean| = {np.abs(dim_means).max():.4f})")
        if not std_check:
            print(f"   Issue: Some dimensions have std far from 1 (max |std-1| = {np.abs(dim_stds - 1.0).max():.4f})")

    return {
        'is_normalized': is_normalized,
        'global_mean': global_mean,
        'global_std': global_std,
        'dim_means': dim_means,
        'dim_stds': dim_stds,
        'sample_means': sample_means,
        'sample_stds': sample_stds
    }


def check_reconstruction_error(original_path, reduced_path, pca_model_path,
                              norm_params_256d_path=None, num_samples=50):
    """
    Check PCA reconstruction error.

    IMPORTANT: If the reduced embeddings are re-normalized, must provide norm_params_256d_path
    to de-normalize them before reconstruction.

    Args:
        original_path: Path to original embeddings
        reduced_path: Path to reduced embeddings
        pca_model_path: Path to PCA model
        norm_params_256d_path: Path to 256d normalization params (if re-normalized)
        num_samples: Number of samples to check

    Returns:
        dict with reconstruction statistics
    """
    print(f"\n{'='*80}")
    print(f"Checking Reconstruction Error")
    print(f"{'='*80}")

    # Load PCA model
    with open(pca_model_path, 'rb') as f:
        pca = pickle.load(f)

    print(f"PCA model: {pca.n_components} components, {pca.n_features_in_} features")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    # Load 256d normalization params if provided
    norm_params_256d = None
    if norm_params_256d_path and Path(norm_params_256d_path).exists():
        norm_params_256d = np.load(norm_params_256d_path)
        print(f"Loaded 256d normalization params: {norm_params_256d_path}")
        print(f"  Will de-normalize before reconstruction")
    else:
        print(f"⚠️  No 256d normalization params provided")
        print(f"  Assuming reduced embeddings are NOT re-normalized")

    # Sample keys
    with h5py.File(original_path, 'r') as f:
        keys = list(f.keys())[:num_samples]

    reconstruction_errors = []
    relative_errors = []

    print(f"\nChecking {num_samples} samples...")
    with h5py.File(original_path, 'r') as f_orig, \
         h5py.File(reduced_path, 'r') as f_red:

        for key in tqdm(keys, desc="Computing errors"):
            original = f_orig[key][:]
            reduced = f_red[key][:]

            # De-normalize 256d if needed
            if norm_params_256d is not None:
                reduced = reduced * norm_params_256d['std'] + norm_params_256d['mean']

            # Reconstruct
            reconstructed = pca.inverse_transform(reduced.reshape(1, -1))

            # Errors
            mse = ((original - reconstructed.squeeze()) ** 2).mean()
            relative = mse / (original ** 2).mean()

            reconstruction_errors.append(mse)
            relative_errors.append(relative)

    reconstruction_errors = np.array(reconstruction_errors)
    relative_errors = np.array(relative_errors)

    print(f"\nReconstruction MSE:")
    print(f"  Mean: {reconstruction_errors.mean():.6f}")
    print(f"  Std: {reconstruction_errors.std():.6f}")
    print(f"  Min: {reconstruction_errors.min():.6f}")
    print(f"  Max: {reconstruction_errors.max():.6f}")

    print(f"\nRelative Error:")
    print(f"  Mean: {relative_errors.mean()*100:.2f}%")
    print(f"  Std: {relative_errors.std()*100:.2f}%")
    print(f"  Min: {relative_errors.min()*100:.2f}%")
    print(f"  Max: {relative_errors.max()*100:.2f}%")

    # Verdict
    if relative_errors.mean() < 0.01:  # < 1% error
        print(f"\n✅ Reconstruction quality is excellent (<1% error)")
    elif relative_errors.mean() < 0.05:  # < 5% error
        print(f"\n✓ Reconstruction quality is good (<5% error)")
    else:
        print(f"\n⚠️  Reconstruction error is significant (>{relative_errors.mean()*100:.1f}%)")

    return {
        'reconstruction_mse': reconstruction_errors,
        'relative_errors': relative_errors,
        'mean_mse': reconstruction_errors.mean(),
        'mean_relative_error': relative_errors.mean()
    }


def check_similarity_preservation(original_path, reduced_path, num_samples=20):
    """
    Check if pairwise similarities are preserved after reduction.

    Args:
        original_path: Path to original embeddings
        reduced_path: Path to reduced embeddings
        num_samples: Number of samples to compare

    Returns:
        dict with similarity preservation statistics
    """
    print(f"\n{'='*80}")
    print(f"Checking Similarity Preservation")
    print(f"{'='*80}")

    # Load samples
    with h5py.File(original_path, 'r') as f:
        keys = list(f.keys())[:num_samples]
        original_embeddings = np.array([f[key][:] for key in keys])

    with h5py.File(reduced_path, 'r') as f:
        reduced_embeddings = np.array([f[key][:] for key in keys])

    print(f"Computing pairwise similarities for {num_samples} samples...")

    # Compute cosine similarities
    def cosine_similarity_matrix(X):
        # Normalize
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        # Compute similarity
        return X_norm @ X_norm.T

    sim_original = cosine_similarity_matrix(original_embeddings)
    sim_reduced = cosine_similarity_matrix(reduced_embeddings)

    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(sim_original, dtype=bool), k=1)
    sim_orig_pairs = sim_original[mask]
    sim_red_pairs = sim_reduced[mask]

    # Correlation
    correlation = np.corrcoef(sim_orig_pairs, sim_red_pairs)[0, 1]

    # Absolute difference
    abs_diff = np.abs(sim_orig_pairs - sim_red_pairs)

    print(f"\nSimilarity Statistics:")
    print(f"  Correlation (original vs reduced): {correlation:.4f}")
    print(f"  Mean absolute difference: {abs_diff.mean():.4f}")
    print(f"  Max absolute difference: {abs_diff.max():.4f}")
    print(f"  Pairs with diff > 0.1: {(abs_diff > 0.1).sum()}/{len(abs_diff)}")

    # Verdict
    if correlation > 0.95:
        print(f"\n✅ Similarity structure is very well preserved (r={correlation:.3f})")
    elif correlation > 0.85:
        print(f"\n✓ Similarity structure is well preserved (r={correlation:.3f})")
    else:
        print(f"\n⚠️  Similarity structure may be degraded (r={correlation:.3f})")

    return {
        'correlation': correlation,
        'mean_abs_diff': abs_diff.mean(),
        'max_abs_diff': abs_diff.max(),
        'sim_original': sim_orig_pairs,
        'sim_reduced': sim_red_pairs
    }


def verify_normalization_params(norm_params_path, embeddings_path):
    """
    Verify that saved normalization parameters are correct.

    Args:
        norm_params_path: Path to normalization parameters
        embeddings_path: Path to embeddings

    Returns:
        bool indicating if params are correct
    """
    print(f"\n{'='*80}")
    print(f"Verifying Normalization Parameters")
    print(f"{'='*80}")

    if not Path(norm_params_path).exists():
        print(f"⚠️  Normalization params not found: {norm_params_path}")
        return False

    # Load params
    params = np.load(norm_params_path)
    saved_mean = params['mean']
    saved_std = params['std']

    print(f"Saved parameters shape: mean={saved_mean.shape}, std={saved_std.shape}")

    # Compute actual stats
    all_embeddings = []
    with h5py.File(embeddings_path, 'r') as f:
        for key in tqdm(list(f.keys()), desc="Loading"):
            all_embeddings.append(f[key][:])

    all_embeddings = np.array(all_embeddings)
    actual_mean = all_embeddings.mean(axis=0)
    actual_std = all_embeddings.std(axis=0)

    print(f"\nComparison:")
    print(f"  Mean difference: {np.abs(saved_mean - actual_mean).mean():.6f}")
    print(f"  Std difference: {np.abs(saved_std - actual_std).mean():.6f}")

    # Check if embeddings are normalized
    is_normalized = (np.abs(actual_mean).mean() < 0.01 and
                     np.abs(actual_std - 1.0).mean() < 0.1)

    if is_normalized:
        print(f"\n✅ Embeddings are normalized, params appear correct")
        return True
    else:
        print(f"\n⚠️  Embeddings do NOT appear normalized")
        print(f"     This could mean:")
        print(f"     1. Normalization was not applied during reduction")
        print(f"     2. Saved params are from a different dataset")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify ligand embedding reduction quality"
    )

    parser.add_argument(
        "--original",
        type=str,
        default="data/processed/ligand_embeddings.h5",
        help="Path to original embeddings"
    )

    parser.add_argument(
        "--reduced",
        type=str,
        default="data/processed/ligand_embeddings_256d.h5",
        help="Path to reduced embeddings"
    )

    parser.add_argument(
        "--pca_model",
        type=str,
        default="data/processed/pca_model_256d.pkl",
        help="Path to PCA model"
    )

    parser.add_argument(
        "--norm_params_256d",
        type=str,
        default="data/processed/ligand_embedding_norm_params_256d.npz",
        help="Path to 256d normalization parameters"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to check (default: 50)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_results",
        help="Directory to save verification plots"
    )

    parser.add_argument(
        "--max_samples_for_norm_check",
        type=int,
        default=None,
        help="Max samples to load for normalization check (None = all). Default: all samples"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("LIGAND EMBEDDING REDUCTION VERIFICATION")
    print("="*80)
    print(f"\nFiles:")
    print(f"  Original: {args.original}")
    print(f"  Reduced: {args.reduced}")
    print(f"  PCA model: {args.pca_model}")
    print(f"  Norm params (256d): {args.norm_params_256d}")
    print(f"\nSettings:")
    print(f"  Num samples for reconstruction/similarity: {args.num_samples}")
    print(f"  Max samples for normalization check: {'All' if args.max_samples_for_norm_check is None else args.max_samples_for_norm_check}")

    results = {}

    # Check 1: Original embeddings normalization
    results['original_norm'] = check_normalization(
        args.original,
        embedding_name="Original (1536d)",
        max_samples=args.max_samples_for_norm_check
    )

    # Check 2: Reduced embeddings normalization
    results['reduced_norm'] = check_normalization(
        args.reduced,
        embedding_name="Reduced (256d)",
        max_samples=args.max_samples_for_norm_check
    )

    # Check 3: Reconstruction error
    results['reconstruction'] = check_reconstruction_error(
        args.original,
        args.reduced,
        args.pca_model,
        norm_params_256d_path=args.norm_params_256d,
        num_samples=args.num_samples
    )

    # Check 4: Similarity preservation
    results['similarity'] = check_similarity_preservation(
        args.original,
        args.reduced,
        num_samples=min(args.num_samples, 20)
    )

    # Check 5: Normalization parameters
    if Path(args.norm_params_256d).exists():
        results['norm_params_valid'] = verify_normalization_params(
            args.norm_params_256d,
            args.reduced
        )

    # Final report
    print(f"\n{'='*80}")
    print("FINAL REPORT")
    print(f"{'='*80}")

    print(f"\n✓ Checks Completed:")
    print(f"  1. Original embeddings normalized: {'✅' if results['original_norm']['is_normalized'] else '❌'}")
    print(f"  2. Reduced embeddings normalized: {'✅' if results['reduced_norm']['is_normalized'] else '❌'}")
    print(f"  3. Reconstruction error: {results['reconstruction']['mean_relative_error']*100:.2f}%")
    print(f"  4. Similarity preservation: r={results['similarity']['correlation']:.3f}")
    if 'norm_params_valid' in results:
        print(f"  5. Normalization params valid: {'✅' if results['norm_params_valid'] else '❌'}")

    # Overall verdict
    all_good = (
        results['original_norm']['is_normalized'] and
        results['reduced_norm']['is_normalized'] and
        results['reconstruction']['mean_relative_error'] < 0.05 and
        results['similarity']['correlation'] > 0.9
    )

    if all_good:
        print(f"\n{'='*80}")
        print(f"🎉 ALL CHECKS PASSED!")
        print(f"{'='*80}")
        print(f"\nThe dimensionality reduction is working correctly:")
        print(f"  ✅ Embeddings are properly normalized")
        print(f"  ✅ Information is well preserved")
        print(f"  ✅ Similarity structure is maintained")
        print(f"\nYou can safely use the reduced embeddings for training!")
    else:
        print(f"\n{'='*80}")
        print(f"⚠️  SOME CHECKS FAILED")
        print(f"{'='*80}")
        print(f"\nPlease review the results above and:")
        print(f"  1. Check if normalization was applied correctly")
        print(f"  2. Verify PCA model is compatible with embeddings")
        print(f"  3. Consider increasing n_components if reconstruction error is high")

    print(f"\n📁 Verification complete!")


if __name__ == "__main__":
    main()
