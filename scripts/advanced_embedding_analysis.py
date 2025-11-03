#!/usr/bin/env python3
"""
Advanced Embedding Analysis

Additional analyses including:
1. Clustering analysis
2. Retrieval performance evaluation
3. Inter/intra-ligand distance analysis
4. Embedding space quality metrics

Usage:
    python scripts/advanced_embedding_analysis.py \\
        --matched_pairs results/visualizations/matched_pairs.json \\
        --output_dir results/advanced_analysis
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine, euclidean
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def load_matched_pairs(json_path):
    """Load matched pairs from JSON file."""
    print(f"Loading matched pairs from {json_path}...")

    with open(json_path, 'r') as f:
        data = json.load(f)

    matched_data = []
    for pocket_id, info in data.items():
        matched_data.append({
            'pocket_id': pocket_id,
            'ligand_name': info['ligand_name'],
            'pocket_embedding': np.array(info['pocket_embedding']),
            'ligand_embedding': np.array(info['ligand_embedding'])
        })

    print(f"Loaded {len(matched_data)} matched pairs")
    return matched_data


def perform_clustering_analysis(matched_data, output_dir):
    """Perform clustering analysis on embeddings."""
    print(f"\n{'='*70}")
    print("Clustering Analysis")
    print(f"{'='*70}\n")

    pocket_embeddings = np.array([d['pocket_embedding'] for d in matched_data])
    ligand_names = [d['ligand_name'] for d in matched_data]

    # Determine optimal number of clusters using elbow method
    print("Finding optimal number of clusters...")
    inertias = []
    silhouette_scores = []
    K_range = range(2, min(21, len(matched_data) // 2))

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pocket_embeddings)
        inertias.append(kmeans.inertia_)

        if k >= 2:
            score = silhouette_score(pocket_embeddings, kmeans.labels_)
            silhouette_scores.append(score)

    # Plot elbow curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(K_range, inertias, 'bo-')
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Inertia', fontsize=12)
    axes[0].set_title('Elbow Method For Optimal k', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(K_range)[1:], silhouette_scores, 'ro-')
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score vs. k', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'clustering_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Use optimal k (based on silhouette score)
    optimal_k = list(K_range)[1:][np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k} (silhouette score: {max(silhouette_scores):.3f})")

    # Perform final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pocket_embeddings)

    # Analyze cluster composition
    cluster_composition = defaultdict(lambda: defaultdict(int))
    for i, label in enumerate(cluster_labels):
        ligand = ligand_names[i]
        cluster_composition[label][ligand] += 1

    print(f"\nCluster composition:")
    for cluster_id in sorted(cluster_composition.keys()):
        ligands = cluster_composition[cluster_id]
        total = sum(ligands.values())
        top_ligands = sorted(ligands.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Cluster {cluster_id} (n={total}): {', '.join([f'{lig}({cnt})' for lig, cnt in top_ligands])}")

    # Visualize clusters
    print("\nVisualizing clusters...")

    # Reduce to 2D for visualization
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(pocket_embeddings)

    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=cluster_labels,
        cmap='tab10',
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidths=0.5
    )

    # Plot cluster centers
    centers_2d = pca.transform(kmeans.cluster_centers_)
    ax.scatter(
        centers_2d[:, 0],
        centers_2d[:, 1],
        c='red',
        marker='X',
        s=500,
        edgecolors='black',
        linewidths=2,
        label='Cluster Centers',
        zorder=10
    )

    ax.set_xlabel('PCA Dimension 1', fontsize=12)
    ax.set_ylabel('PCA Dimension 2', fontsize=12)
    ax.set_title(f'K-Means Clustering (k={optimal_k})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label='Cluster ID')
    plt.tight_layout()
    plt.savefig(output_dir / 'kmeans_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save cluster assignments
    cluster_df = pd.DataFrame({
        'pocket_id': [d['pocket_id'] for d in matched_data],
        'ligand_name': ligand_names,
        'cluster': cluster_labels
    })
    cluster_df.to_csv(output_dir / 'cluster_assignments.csv', index=False)

    print(f"\n✓ Saved cluster assignments to cluster_assignments.csv")

    return cluster_labels, optimal_k


def evaluate_retrieval_performance(matched_data, output_dir):
    """Evaluate retrieval performance (how well pockets retrieve their true ligands)."""
    print(f"\n{'='*70}")
    print("Retrieval Performance Evaluation")
    print(f"{'='*70}\n")

    # Build ligand library (deduplicated)
    ligand_library = {}
    for data in matched_data:
        ligand_name = data['ligand_name']
        if ligand_name not in ligand_library:
            ligand_library[ligand_name] = data['ligand_embedding']

    print(f"Ligand library size: {len(ligand_library)}")

    # For each pocket, rank all ligands by similarity
    retrieval_results = []

    for data in matched_data:
        pocket_emb = data['pocket_embedding']
        true_ligand = data['ligand_name']

        # Compute distances to all ligands
        distances = []
        for ligand_name, ligand_emb in ligand_library.items():
            dist = cosine(pocket_emb, ligand_emb)
            distances.append((ligand_name, dist))

        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[1])

        # Find rank of true ligand
        rank = next(i for i, (lig, _) in enumerate(distances) if lig == true_ligand) + 1

        retrieval_results.append({
            'pocket_id': data['pocket_id'],
            'true_ligand': true_ligand,
            'rank': rank,
            'top1': distances[0][0],
            'top1_distance': distances[0][1],
            'true_ligand_distance': distances[rank - 1][1]
        })

    df_retrieval = pd.DataFrame(retrieval_results)

    # Calculate metrics
    top1_accuracy = (df_retrieval['rank'] == 1).mean()
    top5_accuracy = (df_retrieval['rank'] <= 5).mean()
    top10_accuracy = (df_retrieval['rank'] <= 10).mean()
    mrr = (1.0 / df_retrieval['rank']).mean()  # Mean Reciprocal Rank
    mean_rank = df_retrieval['rank'].mean()
    median_rank = df_retrieval['rank'].median()

    print(f"\nRetrieval Metrics:")
    print(f"  Top-1 Accuracy:  {top1_accuracy:.3f} ({(top1_accuracy*100):.1f}%)")
    print(f"  Top-5 Accuracy:  {top5_accuracy:.3f} ({(top5_accuracy*100):.1f}%)")
    print(f"  Top-10 Accuracy: {top10_accuracy:.3f} ({(top10_accuracy*100):.1f}%)")
    print(f"  Mean Rank:       {mean_rank:.2f}")
    print(f"  Median Rank:     {median_rank:.1f}")
    print(f"  MRR:             {mrr:.3f}")

    # Plot rank distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df_retrieval['rank'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Rank of True Ligand', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of True Ligand Ranks', fontweight='bold')
    axes[0].axvline(mean_rank, color='red', linestyle='--', label=f'Mean: {mean_rank:.1f}')
    axes[0].axvline(median_rank, color='orange', linestyle='--', label=f'Median: {median_rank:.1f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cumulative accuracy curve
    ranks_sorted = sorted(df_retrieval['rank'])
    cumulative_accuracy = [(i + 1) / len(ranks_sorted) for i in range(len(ranks_sorted))]

    axes[1].plot(ranks_sorted, cumulative_accuracy, linewidth=2, color='darkgreen')
    axes[1].set_xlabel('Rank Threshold', fontsize=12)
    axes[1].set_ylabel('Cumulative Accuracy', fontsize=12)
    axes[1].set_title('Cumulative Retrieval Accuracy', fontweight='bold')
    axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50% accuracy')
    axes[1].axvline(median_rank, color='orange', linestyle='--', alpha=0.5, label=f'Median rank: {median_rank:.0f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'retrieval_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save retrieval results
    df_retrieval.to_csv(output_dir / 'retrieval_results.csv', index=False)

    # Analyze failures (where rank > 1)
    failures = df_retrieval[df_retrieval['rank'] > 1]
    if len(failures) > 0:
        print(f"\nRetrieval failures (rank > 1): {len(failures)}")
        print(f"Sample failures:")
        print(failures[['pocket_id', 'true_ligand', 'rank', 'top1']].head(10).to_string(index=False))

    print(f"\n✓ Saved retrieval results to retrieval_results.csv")

    return df_retrieval


def analyze_intra_inter_ligand_distances(matched_data, output_dir):
    """Analyze intra-ligand vs inter-ligand distances."""
    print(f"\n{'='*70}")
    print("Intra-Ligand vs Inter-Ligand Distance Analysis")
    print(f"{'='*70}\n")

    # Group pockets by ligand
    ligand_groups = defaultdict(list)
    for i, data in enumerate(matched_data):
        ligand_groups[data['ligand_name']].append(i)

    # Calculate intra-ligand distances (pockets binding same ligand)
    intra_distances = []
    for ligand_name, indices in ligand_groups.items():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i = indices[i]
                idx_j = indices[j]
                emb_i = matched_data[idx_i]['pocket_embedding']
                emb_j = matched_data[idx_j]['pocket_embedding']
                dist = cosine(emb_i, emb_j)
                intra_distances.append({
                    'ligand': ligand_name,
                    'distance': dist,
                    'type': 'intra'
                })

    # Calculate inter-ligand distances (pockets binding different ligands)
    # Sample to avoid too many comparisons
    np.random.seed(42)
    n_samples = min(10000, len(matched_data) * (len(matched_data) - 1) // 2)
    sampled_pairs = np.random.choice(len(matched_data), size=(n_samples, 2), replace=True)

    inter_distances = []
    for i, j in sampled_pairs:
        if i == j:
            continue
        lig_i = matched_data[i]['ligand_name']
        lig_j = matched_data[j]['ligand_name']

        if lig_i != lig_j:
            emb_i = matched_data[i]['pocket_embedding']
            emb_j = matched_data[j]['pocket_embedding']
            dist = cosine(emb_i, emb_j)
            inter_distances.append({
                'ligand_pair': f"{lig_i}-{lig_j}",
                'distance': dist,
                'type': 'inter'
            })

    print(f"Intra-ligand distances: {len(intra_distances)}")
    print(f"Inter-ligand distances: {len(inter_distances)}")

    # Create DataFrame
    df_intra = pd.DataFrame(intra_distances)
    df_inter = pd.DataFrame(inter_distances)

    if len(df_intra) == 0 or len(df_inter) == 0:
        print("Warning: Not enough data for intra/inter analysis")
        return

    # Calculate statistics
    print(f"\nIntra-ligand distance statistics:")
    print(f"  Mean:   {df_intra['distance'].mean():.4f}")
    print(f"  Median: {df_intra['distance'].median():.4f}")
    print(f"  Std:    {df_intra['distance'].std():.4f}")

    print(f"\nInter-ligand distance statistics:")
    print(f"  Mean:   {df_inter['distance'].mean():.4f}")
    print(f"  Median: {df_inter['distance'].median():.4f}")
    print(f"  Std:    {df_inter['distance'].std():.4f}")

    # Plot distributions
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(df_intra['distance'], bins=50, alpha=0.6, label='Intra-ligand',
           color='blue', edgecolor='black', density=True)
    ax.hist(df_inter['distance'], bins=50, alpha=0.6, label='Inter-ligand',
           color='red', edgecolor='black', density=True)

    ax.axvline(df_intra['distance'].mean(), color='blue', linestyle='--',
              label=f'Intra mean: {df_intra["distance"].mean():.3f}')
    ax.axvline(df_inter['distance'].mean(), color='red', linestyle='--',
              label=f'Inter mean: {df_inter["distance"].mean():.3f}')

    ax.set_xlabel('Cosine Distance', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Intra-Ligand vs Inter-Ligand Distance Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'intra_inter_distances.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Box plot comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    data_to_plot = [df_intra['distance'], df_inter['distance']]
    box = ax.boxplot(data_to_plot, labels=['Intra-ligand', 'Inter-ligand'],
                     patch_artist=True, notch=True, showmeans=True)

    box['boxes'][0].set_facecolor('lightblue')
    box['boxes'][1].set_facecolor('lightcoral')

    ax.set_ylabel('Cosine Distance', fontsize=12)
    ax.set_title('Intra vs Inter-Ligand Distances', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'intra_inter_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save data
    pd.concat([df_intra, df_inter]).to_csv(output_dir / 'intra_inter_distances.csv', index=False)

    print(f"\n✓ Saved distance data to intra_inter_distances.csv")


def create_ligand_similarity_heatmap(matched_data, output_dir, top_n=20):
    """Create heatmap of average distances between ligand types."""
    print(f"\n{'='*70}")
    print(f"Ligand Similarity Heatmap (Top {top_n})")
    print(f"{'='*70}\n")

    # Get top N ligands by frequency
    ligand_counts = {}
    for data in matched_data:
        lig = data['ligand_name']
        ligand_counts[lig] = ligand_counts.get(lig, 0) + 1

    top_ligands = sorted(ligand_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_ligand_names = [lig for lig, _ in top_ligands]

    print(f"Selected top {len(top_ligand_names)} ligands")

    # Group pocket embeddings by ligand
    ligand_embeddings = defaultdict(list)
    for data in matched_data:
        if data['ligand_name'] in top_ligand_names:
            ligand_embeddings[data['ligand_name']].append(data['pocket_embedding'])

    # Compute average embedding for each ligand
    ligand_avg_embeddings = {}
    for lig, embs in ligand_embeddings.items():
        ligand_avg_embeddings[lig] = np.mean(embs, axis=0)

    # Compute pairwise distances
    n_ligands = len(top_ligand_names)
    distance_matrix = np.zeros((n_ligands, n_ligands))

    for i, lig_i in enumerate(top_ligand_names):
        for j, lig_j in enumerate(top_ligand_names):
            if i == j:
                distance_matrix[i, j] = 0
            else:
                dist = cosine(ligand_avg_embeddings[lig_i], ligand_avg_embeddings[lig_j])
                distance_matrix[i, j] = dist

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(distance_matrix, cmap='viridis', aspect='auto')

    # Set ticks
    ax.set_xticks(range(n_ligands))
    ax.set_yticks(range(n_ligands))
    ax.set_xticklabels(top_ligand_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(top_ligand_names, fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Distance', fontsize=12)

    # Add title
    ax.set_title(f'Ligand-Ligand Similarity Heatmap (Top {top_n})\n(Based on Average Pocket Embeddings)',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'ligand_similarity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save distance matrix
    df_dist_matrix = pd.DataFrame(distance_matrix, index=top_ligand_names, columns=top_ligand_names)
    df_dist_matrix.to_csv(output_dir / 'ligand_distance_matrix.csv')

    print(f"\n✓ Saved ligand distance matrix to ligand_distance_matrix.csv")


def hierarchical_clustering_dendrogram(matched_data, output_dir, top_n=30):
    """Create hierarchical clustering dendrogram for ligands."""
    print(f"\n{'='*70}")
    print(f"Hierarchical Clustering Dendrogram (Top {top_n})")
    print(f"{'='*70}\n")

    # Get top N ligands
    ligand_counts = {}
    for data in matched_data:
        lig = data['ligand_name']
        ligand_counts[lig] = ligand_counts.get(lig, 0) + 1

    top_ligands = sorted(ligand_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_ligand_names = [lig for lig, _ in top_ligands]

    # Compute average embeddings
    ligand_embeddings = defaultdict(list)
    for data in matched_data:
        if data['ligand_name'] in top_ligand_names:
            ligand_embeddings[data['ligand_name']].append(data['pocket_embedding'])

    ligand_avg_embeddings = []
    ligand_labels = []
    for lig in top_ligand_names:
        if lig in ligand_embeddings:
            avg_emb = np.mean(ligand_embeddings[lig], axis=0)
            ligand_avg_embeddings.append(avg_emb)
            ligand_labels.append(f"{lig} (n={len(ligand_embeddings[lig])})")

    ligand_avg_embeddings = np.array(ligand_avg_embeddings)

    # Perform hierarchical clustering
    linkage_matrix = linkage(ligand_avg_embeddings, method='ward')

    # Plot dendrogram
    fig, ax = plt.subplots(figsize=(14, 8))

    dendrogram(
        linkage_matrix,
        labels=ligand_labels,
        ax=ax,
        orientation='right',
        distance_sort='ascending',
        color_threshold=0.7 * max(linkage_matrix[:, 2])
    )

    ax.set_xlabel('Distance', fontsize=12)
    ax.set_title(f'Hierarchical Clustering of Ligands (Top {top_n})\nBased on Average Pocket Embeddings',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'ligand_dendrogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Created hierarchical clustering dendrogram")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced embedding analysis"
    )

    parser.add_argument("--matched_pairs", type=str, required=True,
                       help="Path to matched_pairs.json from visualize_embeddings.py")
    parser.add_argument("--output_dir", type=str, default="results/advanced_analysis",
                       help="Output directory for analysis results")

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Advanced Embedding Analysis Pipeline")
    print(f"{'='*70}\n")

    # Load data
    matched_data = load_matched_pairs(args.matched_pairs)

    # 1. Clustering analysis
    cluster_labels, optimal_k = perform_clustering_analysis(matched_data, output_dir)

    # 2. Retrieval performance
    df_retrieval = evaluate_retrieval_performance(matched_data, output_dir)

    # 3. Intra/inter-ligand distances
    analyze_intra_inter_ligand_distances(matched_data, output_dir)

    # 4. Ligand similarity heatmap
    create_ligand_similarity_heatmap(matched_data, output_dir, top_n=20)

    # 5. Hierarchical clustering
    hierarchical_clustering_dendrogram(matched_data, output_dir, top_n=30)

    print(f"\n{'='*70}")
    print("✓ Advanced analysis completed!")
    print(f"{'='*70}\n")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
