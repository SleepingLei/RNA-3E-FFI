#!/usr/bin/env python3
"""
Comprehensive Embedding Visualization Script

This script:
1. Runs inference on all pocket graphs in a directory
2. Loads corresponding ligand embeddings
3. Performs dimensionality reduction (PCA, t-SNE, UMAP)
4. Creates various visualizations and analyses

Usage:
    python scripts/visualize_embeddings.py \\
        --checkpoint models/checkpoints/best_model.pt \\
        --graph_dir data/processed/graphs \\
        --ligand_embeddings data/processed/ligand_embeddings_dedup.h5 \\
        --output_dir results/visualizations
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import h5py
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.deduplicate_ligand_embeddings import extract_ligand_name

# Try to import UMAP (optional)
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not installed. Install with: pip install umap-learn")


def load_splits(splits_file, split_names):
    """
    Load dataset splits from JSON file.

    Args:
        splits_file: Path to splits.json file
        split_names: List of split names to load (e.g., ['train', 'val'])

    Returns:
        Set of complex IDs to include
    """
    if splits_file is None:
        return None

    splits_path = Path(splits_file)
    if not splits_path.exists():
        print(f"Warning: Splits file not found: {splits_file}")
        return None

    try:
        with open(splits_path, 'r') as f:
            splits_data = json.load(f)

        # Collect complex IDs from requested splits
        selected_ids = set()
        for split_name in split_names:
            if split_name in splits_data:
                selected_ids.update(splits_data[split_name])
                print(f"  Loaded {len(splits_data[split_name])} samples from '{split_name}' split")
            else:
                print(f"  Warning: Split '{split_name}' not found in {splits_file}")

        return selected_ids

    except Exception as e:
        print(f"Error loading splits file: {e}")
        return None


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    from scripts.run_inference import load_model as _load_model
    return _load_model(checkpoint_path, device)


def extract_pocket_id_and_ligand(filename):
    """
    Extract pocket ID and ligand name from graph filename.

    Args:
        filename: e.g., "1aju_ARG_model0.pt"

    Returns:
        pocket_id: "1aju_ARG_model0"
        ligand_name: "ARG"
    """
    pocket_id = Path(filename).stem  # Remove .pt extension
    ligand_name = extract_ligand_name(pocket_id)
    return pocket_id, ligand_name


def batch_inference_with_metadata(checkpoint_path, graph_dir, device, selected_ids=None):
    """
    Run inference on all graphs and collect metadata.

    Args:
        checkpoint_path: Path to model checkpoint
        graph_dir: Directory containing graph files
        device: Device to use
        selected_ids: Optional set of complex IDs to filter (from splits)

    Returns:
        Dict with pocket_id as keys, containing:
            - embedding: numpy array
            - ligand_name: str
            - pdb_id: str
            - model_num: int
    """
    print(f"\n{'='*70}")
    print("Step 1: Running Batch Inference on Pocket Graphs")
    print(f"{'='*70}\n")

    from scripts.run_inference import predict_pocket_embedding

    # Load model
    model, config = load_model(checkpoint_path, device)

    # Find all graph files
    graph_files = sorted(Path(graph_dir).glob("*.pt"))
    print(f"Found {len(graph_files)} pocket graphs in {graph_dir}")

    if len(graph_files) == 0:
        raise ValueError(f"No .pt files found in {graph_dir}")

    # Filter by splits if specified
    if selected_ids is not None:
        original_count = len(graph_files)
        graph_files = [f for f in graph_files if f.stem in selected_ids]
        print(f"Filtered to {len(graph_files)} graphs based on split selection (removed {original_count - len(graph_files)})")

    # Predict embeddings
    results = {}

    for graph_file in tqdm(graph_files, desc="Processing pocket graphs"):
        try:
            # Extract metadata
            pocket_id, ligand_name = extract_pocket_id_and_ligand(graph_file.name)

            # Extract PDB ID and model number
            parts = pocket_id.split('_')
            pdb_id = parts[0] if len(parts) > 0 else "unknown"
            model_num = int(parts[-1].replace('model', '')) if 'model' in parts[-1] else 0

            # Load graph
            graph = torch.load(graph_file, weights_only=False)

            # Predict embedding
            embedding = predict_pocket_embedding(graph, model, device)

            results[pocket_id] = {
                'embedding': embedding.flatten(),  # Ensure 1D
                'ligand_name': ligand_name,
                'pdb_id': pdb_id,
                'model_num': model_num,
                'filename': graph_file.name
            }

        except Exception as e:
            print(f"\nError processing {graph_file.name}: {e}")
            continue

    print(f"\nSuccessfully processed {len(results)} pocket embeddings")
    return results


def load_ligand_embeddings(ligand_embeddings_path):
    """Load deduplicated ligand embeddings."""
    print(f"\n{'='*70}")
    print("Step 2: Loading Ligand Embeddings")
    print(f"{'='*70}\n")

    print(f"Loading from {ligand_embeddings_path}...")

    ligand_embeddings = {}

    with h5py.File(ligand_embeddings_path, 'r') as f:
        for ligand_name in f.keys():
            embedding = np.array(f[ligand_name][:])
            ligand_embeddings[ligand_name] = embedding.flatten()  # Ensure 1D

    print(f"Loaded {len(ligand_embeddings)} unique ligand embeddings")
    print(f"Sample ligands: {list(ligand_embeddings.keys())[:10]}")

    return ligand_embeddings


def match_pocket_ligand_pairs(pocket_results, ligand_embeddings):
    """
    Match pocket embeddings with their corresponding ligand embeddings.

    Returns:
        matched_data: List of dicts with matched pairs
        unmatched_pockets: List of pocket IDs without ligand matches
    """
    print(f"\n{'='*70}")
    print("Step 3: Matching Pocket-Ligand Pairs")
    print(f"{'='*70}\n")

    matched_data = []
    unmatched_pockets = []

    for pocket_id, pocket_data in pocket_results.items():
        ligand_name = pocket_data['ligand_name']

        if ligand_name in ligand_embeddings:
            matched_data.append({
                'pocket_id': pocket_id,
                'ligand_name': ligand_name,
                'pdb_id': pocket_data['pdb_id'],
                'model_num': pocket_data['model_num'],
                'pocket_embedding': pocket_data['embedding'],
                'ligand_embedding': ligand_embeddings[ligand_name]
            })
        else:
            unmatched_pockets.append(pocket_id)

    print(f"Matched pairs: {len(matched_data)}")
    print(f"Unmatched pockets: {len(unmatched_pockets)}")

    if unmatched_pockets:
        print(f"\nSample unmatched pockets: {unmatched_pockets[:5]}")
        ligand_names = list(set([pocket_results[p]['ligand_name'] for p in unmatched_pockets[:10]]))
        print(f"Missing ligand names: {ligand_names[:10]}")

    return matched_data, unmatched_pockets


def perform_dimensionality_reduction(embeddings, labels, method='pca', n_components=2, **kwargs):
    """
    Perform dimensionality reduction.

    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        labels: list of labels for each sample
        method: 'pca', 'tsne', or 'umap'
        n_components: number of dimensions to reduce to
        **kwargs: additional parameters for the method

    Returns:
        reduced_embeddings: numpy array of shape (n_samples, n_components)
    """
    print(f"  Applying {method.upper()}...")

    if method == 'pca':
        reducer = PCA(n_components=n_components, **kwargs)
        reduced = reducer.fit_transform(embeddings)
        print(f"    Explained variance: {reducer.explained_variance_ratio_.sum():.3f}")

    elif method == 'tsne':
        default_params = {
            'n_components': n_components,
            'perplexity': min(30, len(embeddings) - 1),
            'random_state': 42,
            'n_iter': 1000
        }
        default_params.update(kwargs)
        reducer = TSNE(**default_params)
        reduced = reducer.fit_transform(embeddings)

    elif method == 'umap':
        if not UMAP_AVAILABLE:
            raise ValueError("UMAP not available. Install with: pip install umap-learn")
        default_params = {
            'n_components': n_components,
            'random_state': 42,
            'n_neighbors': min(15, len(embeddings) - 1),
            'min_dist': 0.1
        }
        default_params.update(kwargs)
        reducer = UMAP(**default_params)
        reduced = reducer.fit_transform(embeddings)

    else:
        raise ValueError(f"Unknown method: {method}")

    return reduced


def visualize_joint_embeddings(matched_data, output_dir, methods=['pca', 'tsne', 'umap']):
    """Create joint visualizations of pocket and ligand embeddings."""
    print(f"\n{'='*70}")
    print("Step 4: Joint Dimensionality Reduction & Visualization")
    print(f"{'='*70}\n")

    # Prepare data
    pocket_embeddings = np.array([d['pocket_embedding'] for d in matched_data])
    ligand_embeddings = np.array([d['ligand_embedding'] for d in matched_data])

    # Combine pocket and ligand embeddings
    all_embeddings = np.vstack([pocket_embeddings, ligand_embeddings])

    # Create labels
    n_pockets = len(pocket_embeddings)
    pocket_labels = [f"{d['pocket_id']}" for d in matched_data]
    ligand_labels = [f"{d['ligand_name']} (ligand)" for d in matched_data]
    all_labels = pocket_labels + ligand_labels

    # Type labels (for coloring)
    type_labels = ['Pocket'] * n_pockets + ['Ligand'] * n_pockets

    # Ligand name labels (for additional coloring)
    ligand_name_labels = [d['ligand_name'] for d in matched_data] * 2

    # Apply each reduction method
    for method in methods:
        if method == 'umap' and not UMAP_AVAILABLE:
            print(f"Skipping {method.upper()} (not available)")
            continue

        try:
            # Reduce to 2D
            reduced_2d = perform_dimensionality_reduction(
                all_embeddings, all_labels, method=method, n_components=2
            )

            # Create DataFrame for plotting
            df = pd.DataFrame({
                f'{method.upper()}_1': reduced_2d[:, 0],
                f'{method.upper()}_2': reduced_2d[:, 1],
                'Type': type_labels,
                'Ligand': ligand_name_labels,
                'Label': all_labels
            })

            # Plot 1: Colored by type (Pocket vs Ligand)
            fig, ax = plt.subplots(figsize=(12, 8))

            for emb_type in ['Pocket', 'Ligand']:
                mask = df['Type'] == emb_type
                ax.scatter(
                    df[mask][f'{method.upper()}_1'],
                    df[mask][f'{method.upper()}_2'],
                    label=emb_type,
                    alpha=0.6,
                    s=100 if emb_type == 'Pocket' else 50,
                    marker='o' if emb_type == 'Pocket' else '^'
                )

            ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
            ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
            ax.set_title(f'Joint Embedding Visualization ({method.upper()})\nPockets vs Ligands',
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / f'joint_{method}_by_type.png', dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / f'joint_{method}_by_type.pdf', bbox_inches='tight')
            plt.close()

            # Plot 2: Colored by ligand name (show top N ligands)
            ligand_counts = pd.Series(ligand_name_labels).value_counts()
            top_ligands = ligand_counts.head(20).index.tolist()

            fig, ax = plt.subplots(figsize=(14, 10))

            # Plot non-top ligands in gray
            mask_other = ~df['Ligand'].isin(top_ligands)
            ax.scatter(
                df[mask_other][f'{method.upper()}_1'],
                df[mask_other][f'{method.upper()}_2'],
                c='lightgray',
                alpha=0.3,
                s=30,
                label='Other ligands'
            )

            # Plot top ligands with distinct colors
            colors = plt.cm.tab20(np.linspace(0, 1, len(top_ligands)))

            for i, ligand in enumerate(top_ligands):
                mask = df['Ligand'] == ligand
                pocket_mask = mask & (df['Type'] == 'Pocket')
                ligand_mask = mask & (df['Type'] == 'Ligand')

                # Plot pockets
                ax.scatter(
                    df[pocket_mask][f'{method.upper()}_1'],
                    df[pocket_mask][f'{method.upper()}_2'],
                    c=[colors[i]],
                    alpha=0.7,
                    s=100,
                    marker='o',
                    label=f'{ligand} (pocket)'
                )

                # Plot ligands
                ax.scatter(
                    df[ligand_mask][f'{method.upper()}_1'],
                    df[ligand_mask][f'{method.upper()}_2'],
                    c=[colors[i]],
                    alpha=0.9,
                    s=150,
                    marker='*',
                    edgecolors='black',
                    linewidths=0.5
                )

            ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
            ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
            ax.set_title(f'Joint Embedding Visualization ({method.upper()})\nTop 20 Ligands',
                        fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / f'joint_{method}_by_ligand.png', dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / f'joint_{method}_by_ligand.pdf', bbox_inches='tight')
            plt.close()

            # Plot 3: Connected pairs (pocket to ligand)
            fig, ax = plt.subplots(figsize=(14, 10))

            # Draw connections between pocket-ligand pairs
            for i in range(n_pockets):
                x_vals = [reduced_2d[i, 0], reduced_2d[i + n_pockets, 0]]
                y_vals = [reduced_2d[i, 1], reduced_2d[i + n_pockets, 1]]
                ax.plot(x_vals, y_vals, 'gray', alpha=0.2, linewidth=0.5, zorder=1)

            # Plot pockets
            ax.scatter(
                reduced_2d[:n_pockets, 0],
                reduced_2d[:n_pockets, 1],
                c='blue',
                alpha=0.6,
                s=100,
                marker='o',
                label='Pocket',
                zorder=2
            )

            # Plot ligands
            ax.scatter(
                reduced_2d[n_pockets:, 0],
                reduced_2d[n_pockets:, 1],
                c='red',
                alpha=0.6,
                s=150,
                marker='^',
                label='Ligand',
                zorder=2
            )

            ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=12)
            ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=12)
            ax.set_title(f'Pocket-Ligand Pair Connections ({method.upper()})',
                        fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / f'joint_{method}_connections.png', dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / f'joint_{method}_connections.pdf', bbox_inches='tight')
            plt.close()

            print(f"  ✓ Created {method.upper()} visualizations")

        except Exception as e:
            print(f"  ✗ Error with {method.upper()}: {e}")
            continue


def analyze_embedding_distances(matched_data, output_dir):
    """Analyze distances between pocket and ligand embeddings."""
    print(f"\n{'='*70}")
    print("Step 5: Distance Analysis")
    print(f"{'='*70}\n")

    distances = {
        'pocket_id': [],
        'ligand_name': [],
        'cosine_distance': [],
        'euclidean_distance': [],
        'cosine_similarity': []
    }

    for data in matched_data:
        pocket_emb = data['pocket_embedding']
        ligand_emb = data['ligand_embedding']

        # Calculate distances
        cos_dist = cosine(pocket_emb, ligand_emb)
        euc_dist = euclidean(pocket_emb, ligand_emb)
        cos_sim = 1 - cos_dist

        distances['pocket_id'].append(data['pocket_id'])
        distances['ligand_name'].append(data['ligand_name'])
        distances['cosine_distance'].append(cos_dist)
        distances['euclidean_distance'].append(euc_dist)
        distances['cosine_similarity'].append(cos_sim)

    df_dist = pd.DataFrame(distances)

    # Print statistics
    print("\nDistance Statistics:")
    print(df_dist[['cosine_distance', 'euclidean_distance', 'cosine_similarity']].describe())

    # Plot distance distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Cosine distance
    axes[0].hist(df_dist['cosine_distance'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Cosine Distance', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Pocket-Ligand Cosine Distance Distribution', fontweight='bold')
    axes[0].axvline(df_dist['cosine_distance'].mean(), color='red',
                    linestyle='--', label=f'Mean: {df_dist["cosine_distance"].mean():.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Euclidean distance
    axes[1].hist(df_dist['euclidean_distance'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('Euclidean Distance', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Pocket-Ligand Euclidean Distance Distribution', fontweight='bold')
    axes[1].axvline(df_dist['euclidean_distance'].mean(), color='red',
                    linestyle='--', label=f'Mean: {df_dist["euclidean_distance"].mean():.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Cosine similarity
    axes[2].hist(df_dist['cosine_similarity'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[2].set_xlabel('Cosine Similarity', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('Pocket-Ligand Cosine Similarity Distribution', fontweight='bold')
    axes[2].axvline(df_dist['cosine_similarity'].mean(), color='red',
                    linestyle='--', label=f'Mean: {df_dist["cosine_similarity"].mean():.3f}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'distance_distributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'distance_distributions.pdf', bbox_inches='tight')
    plt.close()

    # Save distance data
    df_dist.to_csv(output_dir / 'pocket_ligand_distances.csv', index=False)
    print(f"\n✓ Saved distance data to pocket_ligand_distances.csv")

    # Analyze by ligand type
    ligand_stats = df_dist.groupby('ligand_name').agg({
        'cosine_distance': ['mean', 'std', 'count'],
        'cosine_similarity': ['mean', 'std']
    }).round(4)

    ligand_stats.columns = ['_'.join(col).strip() for col in ligand_stats.columns.values]
    ligand_stats = ligand_stats.sort_values('cosine_distance_mean')

    print("\nTop 10 Ligands by Average Cosine Distance (most similar):")
    print(ligand_stats.head(10))

    ligand_stats.to_csv(output_dir / 'ligand_distance_stats.csv')

    return df_dist


def analyze_embedding_correlation(matched_data, output_dir):
    """Analyze correlation between pocket and ligand embeddings."""
    print(f"\n{'='*70}")
    print("Step 6: Embedding Correlation Analysis")
    print(f"{'='*70}\n")

    pocket_embeddings = np.array([d['pocket_embedding'] for d in matched_data])
    ligand_embeddings = np.array([d['ligand_embedding'] for d in matched_data])

    # Compute pairwise correlations
    correlations = []

    for i in range(len(matched_data)):
        pearson_r, _ = pearsonr(pocket_embeddings[i], ligand_embeddings[i])
        spearman_r, _ = spearmanr(pocket_embeddings[i], ligand_embeddings[i])

        correlations.append({
            'pocket_id': matched_data[i]['pocket_id'],
            'ligand_name': matched_data[i]['ligand_name'],
            'pearson_r': pearson_r,
            'spearman_r': spearman_r
        })

    df_corr = pd.DataFrame(correlations)

    print("\nCorrelation Statistics:")
    print(df_corr[['pearson_r', 'spearman_r']].describe())

    # Plot correlation distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df_corr['pearson_r'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Pearson Correlation', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Pocket-Ligand Pearson Correlation Distribution', fontweight='bold')
    axes[0].axvline(df_corr['pearson_r'].mean(), color='red',
                   linestyle='--', label=f'Mean: {df_corr["pearson_r"].mean():.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(df_corr['spearman_r'], bins=50, alpha=0.7, color='coral', edgecolor='black')
    axes[1].set_xlabel('Spearman Correlation', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Pocket-Ligand Spearman Correlation Distribution', fontweight='bold')
    axes[1].axvline(df_corr['spearman_r'].mean(), color='red',
                   linestyle='--', label=f'Mean: {df_corr["spearman_r"].mean():.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_distributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'correlation_distributions.pdf', bbox_inches='tight')
    plt.close()

    df_corr.to_csv(output_dir / 'pocket_ligand_correlations.csv', index=False)
    print(f"\n✓ Saved correlation data to pocket_ligand_correlations.csv")

    return df_corr


def create_ligand_summary(matched_data, output_dir):
    """Create summary statistics by ligand type."""
    print(f"\n{'='*70}")
    print("Step 7: Ligand Summary Statistics")
    print(f"{'='*70}\n")

    # Count pockets per ligand
    ligand_counts = {}
    for data in matched_data:
        ligand_name = data['ligand_name']
        ligand_counts[ligand_name] = ligand_counts.get(ligand_name, 0) + 1

    # Create summary DataFrame
    summary_data = []
    for ligand_name, count in ligand_counts.items():
        summary_data.append({
            'ligand_name': ligand_name,
            'n_pockets': count
        })

    df_summary = pd.DataFrame(summary_data).sort_values('n_pockets', ascending=False)

    print(f"\nTotal unique ligands: {len(df_summary)}")
    print(f"\nTop 20 ligands by pocket count:")
    print(df_summary.head(20).to_string(index=False))

    # Plot ligand distribution
    fig, ax = plt.subplots(figsize=(12, 6))

    top_n = 30
    df_top = df_summary.head(top_n)

    ax.bar(range(len(df_top)), df_top['n_pockets'], color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(df_top)))
    ax.set_xticklabels(df_top['ligand_name'], rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Ligand Name', fontsize=12)
    ax.set_ylabel('Number of Pockets', fontsize=12)
    ax.set_title(f'Top {top_n} Ligands by Pocket Count', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'ligand_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ligand_distribution.pdf', bbox_inches='tight')
    plt.close()

    df_summary.to_csv(output_dir / 'ligand_summary.csv', index=False)
    print(f"\n✓ Saved ligand summary to ligand_summary.csv")

    return df_summary


def save_embeddings(pocket_results, ligand_embeddings, matched_data, output_dir):
    """Save all embeddings to files."""
    print(f"\n{'='*70}")
    print("Step 8: Saving Embeddings")
    print(f"{'='*70}\n")

    # Save pocket embeddings
    pocket_emb_dict = {k: v['embedding'] for k, v in pocket_results.items()}
    np.savez(output_dir / 'pocket_embeddings.npz', **pocket_emb_dict)
    print(f"✓ Saved {len(pocket_emb_dict)} pocket embeddings to pocket_embeddings.npz")

    # Save matched pairs
    matched_pairs = {}
    for data in matched_data:
        key = data['pocket_id']
        matched_pairs[key] = {
            'pocket_embedding': data['pocket_embedding'].tolist(),
            'ligand_embedding': data['ligand_embedding'].tolist(),
            'ligand_name': data['ligand_name']
        }

    with open(output_dir / 'matched_pairs.json', 'w') as f:
        json.dump(matched_pairs, f, indent=2)
    print(f"✓ Saved {len(matched_pairs)} matched pairs to matched_pairs.json")


def generate_report(matched_data, df_dist, df_corr, df_summary, output_dir):
    """Generate a comprehensive analysis report."""
    print(f"\n{'='*70}")
    print("Step 9: Generating Analysis Report")
    print(f"{'='*70}\n")

    report = []
    report.append("# Pocket-Ligand Embedding Analysis Report\n")
    report.append(f"Generated: {pd.Timestamp.now()}\n\n")

    report.append("## Summary Statistics\n\n")
    report.append(f"- **Total matched pocket-ligand pairs**: {len(matched_data)}\n")
    report.append(f"- **Unique ligands**: {len(df_summary)}\n")
    report.append(f"- **Embedding dimension**: {matched_data[0]['pocket_embedding'].shape[0]}\n\n")

    report.append("## Distance Metrics\n\n")
    report.append("### Cosine Distance\n")
    report.append(f"- Mean: {df_dist['cosine_distance'].mean():.4f}\n")
    report.append(f"- Median: {df_dist['cosine_distance'].median():.4f}\n")
    report.append(f"- Std: {df_dist['cosine_distance'].std():.4f}\n")
    report.append(f"- Min: {df_dist['cosine_distance'].min():.4f}\n")
    report.append(f"- Max: {df_dist['cosine_distance'].max():.4f}\n\n")

    report.append("### Cosine Similarity\n")
    report.append(f"- Mean: {df_dist['cosine_similarity'].mean():.4f}\n")
    report.append(f"- Median: {df_dist['cosine_similarity'].median():.4f}\n")
    report.append(f"- Std: {df_dist['cosine_similarity'].std():.4f}\n\n")

    report.append("## Correlation Analysis\n\n")
    report.append(f"- **Mean Pearson r**: {df_corr['pearson_r'].mean():.4f}\n")
    report.append(f"- **Mean Spearman r**: {df_corr['spearman_r'].mean():.4f}\n\n")

    report.append("## Top 10 Most Common Ligands\n\n")
    report.append(df_summary.head(10).to_markdown(index=False))
    report.append("\n\n")

    report.append("## Output Files\n\n")
    report.append("### Data Files\n")
    report.append("- `pocket_embeddings.npz`: All pocket embeddings\n")
    report.append("- `matched_pairs.json`: Matched pocket-ligand pairs with embeddings\n")
    report.append("- `pocket_ligand_distances.csv`: Distance metrics for each pair\n")
    report.append("- `pocket_ligand_correlations.csv`: Correlation metrics for each pair\n")
    report.append("- `ligand_summary.csv`: Summary statistics by ligand\n")
    report.append("- `ligand_distance_stats.csv`: Distance statistics by ligand type\n\n")

    report.append("### Visualization Files\n")
    report.append("- `joint_pca_*.png/pdf`: PCA visualizations\n")
    report.append("- `joint_tsne_*.png/pdf`: t-SNE visualizations\n")
    report.append("- `joint_umap_*.png/pdf`: UMAP visualizations (if available)\n")
    report.append("- `distance_distributions.png/pdf`: Distance distribution plots\n")
    report.append("- `correlation_distributions.png/pdf`: Correlation distribution plots\n")
    report.append("- `ligand_distribution.png/pdf`: Ligand frequency distribution\n")

    report_text = ''.join(report)

    with open(output_dir / 'analysis_report.md', 'w') as f:
        f.write(report_text)

    print("✓ Generated analysis_report.md")
    print(f"\nReport preview:\n{'-'*70}\n{report_text[:500]}...\n{'-'*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive embedding visualization and analysis"
    )

    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--graph_dir", type=str, required=True,
                       help="Directory containing pocket graph files (.pt)")
    parser.add_argument("--ligand_embeddings", type=str, required=True,
                       help="Path to deduplicated ligand embeddings (HDF5)")

    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="results/visualizations",
                       help="Output directory for visualizations and analysis")
    parser.add_argument("--methods", nargs='+', default=['pca', 'tsne', 'umap'],
                       choices=['pca', 'tsne', 'umap'],
                       help="Dimensionality reduction methods to use")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu, default: auto-detect)")
    parser.add_argument("--splits_file", type=str, default=None,
                       help="Path to splits.json file (e.g., data/splits/splits.json)")
    parser.add_argument("--splits", nargs='+', default=None,
                       choices=['train', 'val', 'test'],
                       help="Which splits to analyze (e.g., train val test). If not specified, all data is used.")

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Pocket-Ligand Embedding Visualization Pipeline")
    print(f"{'='*70}\n")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Graph directory: {args.graph_dir}")
    print(f"Ligand embeddings: {args.ligand_embeddings}")
    print(f"Output directory: {output_dir}")
    print(f"Methods: {', '.join(args.methods)}")

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load splits if specified
    selected_ids = None
    if args.splits_file and args.splits:
        print(f"\nLoading data splits from: {args.splits_file}")
        print(f"Selected splits: {', '.join(args.splits)}")
        selected_ids = load_splits(args.splits_file, args.splits)
        if selected_ids:
            print(f"Total samples selected: {len(selected_ids)}")
    elif args.splits and not args.splits_file:
        print("\nWarning: --splits specified but --splits_file not provided. Ignoring split selection.")
    else:
        print(f"\nNo split selection specified. Using all available data.")

    print()

    # Step 1: Run inference on all pockets
    pocket_results = batch_inference_with_metadata(
        args.checkpoint,
        args.graph_dir,
        device,
        selected_ids=selected_ids
    )

    # Step 2: Load ligand embeddings
    ligand_embeddings = load_ligand_embeddings(args.ligand_embeddings)

    # Step 3: Match pocket-ligand pairs
    matched_data, unmatched = match_pocket_ligand_pairs(pocket_results, ligand_embeddings)

    if len(matched_data) == 0:
        print("\nError: No matched pocket-ligand pairs found!")
        return

    # Step 4: Joint visualization
    visualize_joint_embeddings(matched_data, output_dir, methods=args.methods)

    # Step 5: Distance analysis
    df_dist = analyze_embedding_distances(matched_data, output_dir)

    # Step 6: Correlation analysis
    df_corr = analyze_embedding_correlation(matched_data, output_dir)

    # Step 7: Ligand summary
    df_summary = create_ligand_summary(matched_data, output_dir)

    # Step 8: Save embeddings
    save_embeddings(pocket_results, ligand_embeddings, matched_data, output_dir)

    # Step 9: Generate report
    generate_report(matched_data, df_dist, df_corr, df_summary, output_dir)

    print(f"\n{'='*70}")
    print("✓ Pipeline completed successfully!")
    print(f"{'='*70}\n")
    print(f"All results saved to: {output_dir}")
    print(f"\nKey outputs:")
    print(f"  - Visualizations: {len(args.methods)} methods × 3 plot types")
    print(f"  - Data files: 6 CSV/JSON/NPZ files")
    print(f"  - Report: analysis_report.md")


if __name__ == "__main__":
    main()
