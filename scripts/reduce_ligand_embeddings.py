#!/usr/bin/env python3
"""
Reduce ligand embeddings dimensionality using PCA.

This script reduces the 1536-dimensional ligand embeddings to 256 dimensions
while preserving >99% of the variance (as shown by PCA analysis).

Usage:
    python scripts/reduce_ligand_embeddings.py \
        --input data/processed/ligand_embeddings.h5 \
        --output data/processed/ligand_embeddings_256d.h5 \
        --n_components 256
"""

import argparse
import h5py
import numpy as np
from sklearn.decomposition import PCA
import pickle
from pathlib import Path
from tqdm import tqdm


def reduce_embeddings(input_path, output_path, n_components=256, save_pca_model=True,
                     renormalize_output=True):
    """
    Reduce ligand embeddings using PCA.

    IMPORTANT: This function expects normalized input embeddings (z-score normalized).
    It will apply PCA in the normalized space and optionally re-normalize the output.

    Args:
        input_path: Path to original embeddings HDF5 file (should be normalized)
        output_path: Path to save reduced embeddings
        n_components: Target dimensionality
        save_pca_model: Whether to save the PCA model for later use
        renormalize_output: Whether to re-normalize the reduced embeddings

    Returns:
        Dictionary with reduction statistics
    """
    print("="*80)
    print(f"Reducing Ligand Embeddings: 1536 → {n_components}")
    print("="*80)

    # Load all embeddings
    print(f"\n[1/5] Loading embeddings from: {input_path}")
    embeddings_dict = {}
    keys = []

    with h5py.File(input_path, 'r') as f:
        print(f"Found {len(f.keys())} ligand embeddings")
        for key in tqdm(f.keys(), desc="Loading"):
            embeddings_dict[key] = f[key][:]
            keys.append(key)

    # Convert to array for PCA
    embeddings_array = np.array([embeddings_dict[k] for k in keys])
    original_dim = embeddings_array.shape[1]
    n_samples = embeddings_array.shape[0]

    print(f"\nOriginal shape: {embeddings_array.shape}")
    print(f"  Samples: {n_samples}")
    print(f"  Dimensions: {original_dim}")

    # Check if input is normalized
    print(f"\n[2/5] Checking normalization status...")
    input_mean = embeddings_array.mean(axis=0).mean()
    input_std = embeddings_array.std(axis=0).mean()
    print(f"  Input mean (avg across dims): {input_mean:.6f}")
    print(f"  Input std (avg across dims): {input_std:.6f}")

    if abs(input_mean) > 0.1 or abs(input_std - 1.0) > 0.2:
        print(f"  ⚠️  WARNING: Input embeddings may not be normalized!")
        print(f"     Expected: mean≈0, std≈1")
        print(f"     Got: mean={input_mean:.4f}, std={input_std:.4f}")
        print(f"     PCA will still work, but results may differ from expectations.")
    else:
        print(f"  ✓ Input embeddings are normalized (z-score)")

    # Fit PCA
    print(f"\n[3/5] Fitting PCA with {n_components} components...")
    print(f"  Note: Applying PCA on normalized space (correlation-based PCA)")
    pca = PCA(n_components=n_components)
    embeddings_reduced = pca.fit_transform(embeddings_array)

    # Statistics
    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"\nReduced shape: {embeddings_reduced.shape}")
    print(f"Variance explained: {variance_explained*100:.2f}%")
    print(f"Dimension reduction: {original_dim} → {n_components} " +
          f"({(1-n_components/original_dim)*100:.1f}% reduction)")

    # Top components
    print(f"\nTop 10 components variance:")
    for i in range(min(10, n_components)):
        var = pca.explained_variance_ratio_[i]
        cumvar = pca.explained_variance_ratio_[:i+1].sum()
        print(f"  PC{i+1}: {var*100:.2f}% (cumulative: {cumvar*100:.2f}%)")

    # Re-normalize the reduced embeddings
    norm_params_256d = None
    if renormalize_output:
        print(f"\n[4/5] Re-normalizing {n_components}d embeddings...")
        output_mean = embeddings_reduced.mean(axis=0, keepdims=True)
        output_std = embeddings_reduced.std(axis=0, keepdims=True)
        # Add small epsilon to avoid division by zero
        output_std = np.where(output_std < 1e-8, 1.0, output_std)

        # Check normalization status before
        print(f"  Before re-normalization:")
        print(f"    Mean (avg): {embeddings_reduced.mean(axis=0).mean():.6f}")
        print(f"    Std (avg): {embeddings_reduced.std(axis=0).mean():.6f}")

        # Apply normalization
        embeddings_reduced = (embeddings_reduced - output_mean) / output_std

        # Check after
        print(f"  After re-normalization:")
        print(f"    Mean (avg): {embeddings_reduced.mean(axis=0).mean():.6f}")
        print(f"    Std (avg): {embeddings_reduced.std(axis=0).mean():.6f}")
        print(f"  ✓ {n_components}d embeddings are now normalized")

        norm_params_256d = {
            'mean': output_mean.squeeze(),
            'std': output_std.squeeze()
        }
    else:
        print(f"\n[4/5] Skipping re-normalization (renormalize_output=False)")

    # Save reduced embeddings
    print(f"\n[5/5] Saving reduced embeddings to: {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        for idx, key in enumerate(tqdm(keys, desc="Saving")):
            f.create_dataset(key, data=embeddings_reduced[idx])

    print(f"✓ Saved {len(keys)} reduced embeddings")

    # Save PCA model and normalization parameters
    pca_model_path = None
    if save_pca_model:
        pca_model_path = output_path.parent / f"pca_model_{n_components}d.pkl"
        print(f"\n[6/6] Saving PCA model and parameters...")
        print(f"  PCA model: {pca_model_path}")
        with open(pca_model_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"  ✓ PCA model saved")

        # Save normalization parameters for the reduced space
        if renormalize_output and norm_params_256d is not None:
            norm_path = output_path.parent / f"ligand_embedding_norm_params_{n_components}d.npz"
            print(f"  Normalization params: {norm_path}")
            np.savez(norm_path,
                    mean=norm_params_256d['mean'],
                    std=norm_params_256d['std'])
            print(f"  ✓ {n_components}d normalization parameters saved")
            print(f"    Note: Use these params when normalizing NEW {n_components}d embeddings")

        # Also save transformation info
        info_path = output_path.parent / f"pca_info_{n_components}d.txt"
        with open(info_path, 'w') as f:
            f.write(f"PCA Dimensionality Reduction\n")
            f.write(f"="*80 + "\n\n")
            f.write(f"Original dimension: {original_dim}\n")
            f.write(f"Reduced dimension: {n_components}\n")
            f.write(f"Number of samples: {n_samples}\n")
            f.write(f"Variance explained: {variance_explained*100:.2f}%\n")
            f.write(f"Re-normalized: {'Yes' if renormalize_output else 'No'}\n\n")
            f.write(f"Top 20 components:\n")
            for i in range(min(20, n_components)):
                var = pca.explained_variance_ratio_[i]
                cumvar = pca.explained_variance_ratio_[:i+1].sum()
                f.write(f"  PC{i+1}: {var*100:.4f}% (cumulative: {cumvar*100:.2f}%)\n")
        print(f"  ✓ Info saved to: {info_path}")

    # Verification
    print(f"\n--- Verification ---")
    with h5py.File(output_path, 'r') as f:
        test_key = keys[0]
        test_embedding = f[test_key][:]
        print(f"Sample embedding '{test_key}':")
        print(f"  Shape: {test_embedding.shape}")
        print(f"  Mean: {test_embedding.mean():.4f}")
        print(f"  Std: {test_embedding.std():.4f}")
        print(f"  L2 norm: {np.linalg.norm(test_embedding):.4f}")

    print("\n" + "="*80)
    print("✓ Dimensionality reduction complete!")
    print("="*80)

    return {
        'original_dim': original_dim,
        'reduced_dim': n_components,
        'n_samples': n_samples,
        'variance_explained': variance_explained,
        'pca_model_path': pca_model_path,
        'output_path': output_path,
        'renormalized': renormalize_output,
        'norm_params_path': output_path.parent / f"ligand_embedding_norm_params_{n_components}d.npz" if renormalize_output else None
    }


def apply_pca_to_new_embedding(embedding_1536d, pca_model_path,
                              norm_params_1536d_path=None,
                              norm_params_256d_path=None):
    """
    Apply saved PCA model to a new embedding (for inference).

    Complete pipeline:
    1. Normalize the 1536d embedding (using training set params)
    2. Apply PCA reduction
    3. Normalize the 256d embedding (using training set params)

    Args:
        embedding_1536d: New embedding array [1536] (raw, unnormalized)
        pca_model_path: Path to saved PCA model
        norm_params_1536d_path: Path to 1536d normalization params (optional)
        norm_params_256d_path: Path to 256d normalization params (optional)

    Returns:
        Reduced embedding [256] (normalized if params provided)
    """
    embedding = np.array(embedding_1536d)
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    # Step 1: Normalize 1536d (if params provided)
    if norm_params_1536d_path is not None:
        params_1536d = np.load(norm_params_1536d_path)
        embedding = (embedding - params_1536d['mean']) / params_1536d['std']
        print(f"Applied 1536d normalization")

    # Step 2: Apply PCA
    with open(pca_model_path, 'rb') as f:
        pca = pickle.load(f)
    reduced = pca.transform(embedding)

    # Step 3: Normalize 256d (if params provided)
    if norm_params_256d_path is not None:
        params_256d = np.load(norm_params_256d_path)
        reduced = (reduced - params_256d['mean']) / params_256d['std']
        print(f"Applied 256d normalization")

    return reduced.squeeze()


def main():
    parser = argparse.ArgumentParser(
        description="Reduce ligand embedding dimensionality using PCA"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/ligand_embeddings.h5",
        help="Input HDF5 file with original embeddings"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/ligand_embeddings_256d.h5",
        help="Output HDF5 file for reduced embeddings"
    )

    parser.add_argument(
        "--n_components",
        type=int,
        default=256,
        help="Target dimensionality (default: 256)"
    )

    parser.add_argument(
        "--no_save_pca",
        action="store_true",
        help="Don't save PCA model"
    )

    parser.add_argument(
        "--no_renormalize",
        action="store_true",
        help="Don't re-normalize the reduced embeddings (default: re-normalize)"
    )

    args = parser.parse_args()

    # Run reduction
    results = reduce_embeddings(
        input_path=args.input,
        output_path=args.output,
        n_components=args.n_components,
        save_pca_model=not args.no_save_pca,
        renormalize_output=not args.no_renormalize
    )

    # Print summary
    print(f"\nSummary:")
    print(f"  Input: {args.input}")
    print(f"  Output: {results['output_path']}")
    print(f"  Dimension: {results['original_dim']} → {results['reduced_dim']}")
    print(f"  Samples: {results['n_samples']}")
    print(f"  Variance preserved: {results['variance_explained']*100:.2f}%")
    print(f"  Re-normalized: {'Yes' if results['renormalized'] else 'No'}")
    if results['pca_model_path']:
        print(f"  PCA model: {results['pca_model_path']}")
    if results['norm_params_path']:
        print(f"  Norm params (256d): {results['norm_params_path']}")

    print(f"\n📁 Generated Files:")
    print(f"  {results['output_path']}")
    if results['pca_model_path']:
        print(f"  {results['pca_model_path']}")
    if results['norm_params_path']:
        print(f"  {results['norm_params_path']}")

    print(f"\n🎯 Next Steps:")
    print(f"  1. Update training script to use: {results['output_path']}")
    print(f"  2. Update model output_dim to: {results['reduced_dim']}")
    print(f"  3. For new ligands during inference:")
    print(f"     a. Generate 1536d embedding (Uni-Mol)")
    print(f"     b. Normalize using: ligand_embedding_norm_params.npz")
    print(f"     c. Apply PCA using: {results['pca_model_path']}")
    if results['renormalized']:
        print(f"     d. Normalize using: {results['norm_params_path']}")

    print("\n💻 Example Training Command:")
    print(f"  python scripts/04_train_model.py \\")
    print(f"    --embeddings_path {results['output_path']} \\")
    print(f"    --output_dim {results['reduced_dim']} \\")
    print(f"    --hidden_irreps \"32x0e + 16x1o + 8x2e\" \\")
    print(f"    --num_layers 4 \\")
    print(f"    --batch_size 16 \\")
    print(f"    --num_epochs 300")

    print("\n🔍 Example Inference Code:")
    print(f"  from scripts.reduce_ligand_embeddings import apply_pca_to_new_embedding")
    print(f"  ")
    print(f"  # Generate embedding for new ligand")
    print(f"  embedding_1536d = unimol_model.get_embedding(ligand)")
    print(f"  ")
    print(f"  # Reduce to 256d with normalization")
    print(f"  embedding_256d = apply_pca_to_new_embedding(")
    print(f"      embedding_1536d,")
    print(f"      pca_model_path='{results['pca_model_path']}',")
    print(f"      norm_params_1536d_path='data/processed/ligand_embedding_norm_params.npz',")
    if results['renormalized']:
        print(f"      norm_params_256d_path='{results['norm_params_path']}'")
    print(f"  )")


if __name__ == "__main__":
    main()
