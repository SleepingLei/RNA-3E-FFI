#!/usr/bin/env python3
"""
Example: How to read and work with graph files

This script demonstrates various ways to load and inspect graph files.
"""
import torch
from pathlib import Path


def basic_read_example():
    """Basic example of reading a graph file."""
    print("\n" + "="*70)
    print("Example 1: Basic Graph Reading")
    print("="*70 + "\n")

    # Load a graph file
    graph_path = "data/processed/graphs/1aju_ARG.pt"
    data = torch.load(graph_path)

    print(f"Loaded graph from: {graph_path}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Node feature dimension: {data.x.shape[1]}")
    print(f"Node positions shape: {data.pos.shape}")
    print()


def access_node_features():
    """Example of accessing node features."""
    print("\n" + "="*70)
    print("Example 2: Accessing Node Features")
    print("="*70 + "\n")

    graph_path = "data/processed/graphs/1aju_ARG.pt"
    data = torch.load(graph_path)

    # Access all node features
    node_features = data.x  # Shape: [num_nodes, feature_dim]
    print(f"All node features shape: {node_features.shape}")

    # Access features for a specific node (e.g., node 0)
    node_0_features = data.x[0]
    print(f"\nNode 0 features: {node_0_features}")
    print(f"Feature dimension: {len(node_0_features)}")

    # Access specific feature dimension for all nodes
    first_feature = data.x[:, 0]  # First feature of all nodes
    print(f"\nFirst feature of all nodes (shape: {first_feature.shape}):")
    print(f"Min: {first_feature.min():.2f}, Max: {first_feature.max():.2f}")

    # Node positions
    node_positions = data.pos  # Shape: [num_nodes, 3] (x, y, z)
    print(f"\nNode positions shape: {node_positions.shape}")
    print(f"Position of node 0: {node_positions[0]}")
    print()


def access_edges():
    """Example of accessing edge information."""
    print("\n" + "="*70)
    print("Example 3: Accessing Edge Information")
    print("="*70 + "\n")

    graph_path = "data/processed/graphs/1aju_ARG.pt"
    data = torch.load(graph_path)

    # Edge indices: [2, num_edges]
    # First row: source nodes, Second row: target nodes
    edge_index = data.edge_index

    print(f"Edge index shape: {edge_index.shape}")
    print(f"Number of edges: {edge_index.shape[1]}")

    # Get first 5 edges
    print("\nFirst 5 edges (source -> target):")
    for i in range(5):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        print(f"  Edge {i}: Node {src} -> Node {dst}")

    # Find neighbors of a specific node
    node_id = 0
    neighbors = edge_index[1][edge_index[0] == node_id]
    print(f"\nNode {node_id} has {len(neighbors)} neighbors:")
    print(f"Neighbor IDs: {neighbors[:10].tolist()}...")  # Show first 10
    print()


def calculate_distances():
    """Example of calculating distances between nodes."""
    print("\n" + "="*70)
    print("Example 4: Calculating Distances")
    print("="*70 + "\n")

    graph_path = "data/processed/graphs/1aju_ARG.pt"
    data = torch.load(graph_path)

    # Calculate distance between node 0 and node 1
    pos_0 = data.pos[0]
    pos_1 = data.pos[1]
    distance = torch.norm(pos_1 - pos_0)

    print(f"Position of node 0: {pos_0}")
    print(f"Position of node 1: {pos_1}")
    print(f"Distance between node 0 and 1: {distance:.4f} Å")

    # Calculate all pairwise distances (for small graphs)
    if data.num_nodes < 100:
        from torch.cdist import cdist
        # Note: For large graphs, this can be memory-intensive
        distances = torch.cdist(data.pos, data.pos)
        print(f"\nPairwise distance matrix shape: {distances.shape}")
        print(f"Average distance: {distances.mean():.4f} Å")
        print(f"Max distance: {distances.max():.4f} Å")
    print()


def batch_loading():
    """Example of loading multiple graphs."""
    print("\n" + "="*70)
    print("Example 5: Loading Multiple Graphs")
    print("="*70 + "\n")

    graph_dir = Path("data/processed/graphs")
    graph_files = list(graph_dir.glob("*.pt"))[:5]  # Load first 5

    print(f"Loading {len(graph_files)} graphs...\n")

    for graph_file in graph_files:
        data = torch.load(graph_file)
        print(f"{graph_file.name}:")
        print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}")

    print()


def advanced_analysis():
    """Example of advanced graph analysis."""
    print("\n" + "="*70)
    print("Example 6: Advanced Graph Analysis")
    print("="*70 + "\n")

    graph_path = "data/processed/graphs/1aju_ARG.pt"
    data = torch.load(graph_path)

    # Calculate node degrees
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    # In-degree: number of incoming edges for each node
    in_degree = torch.zeros(num_nodes, dtype=torch.long)
    for target in edge_index[1]:
        in_degree[target] += 1

    print(f"Node degree statistics:")
    print(f"  Min degree: {in_degree.min()}")
    print(f"  Max degree: {in_degree.max()}")
    print(f"  Average degree: {in_degree.float().mean():.2f}")

    # Find nodes with highest degree (most connected)
    top_k = 5
    top_degrees, top_indices = torch.topk(in_degree, top_k)
    print(f"\nTop {top_k} most connected nodes:")
    for i, (idx, deg) in enumerate(zip(top_indices, top_degrees)):
        print(f"  {i+1}. Node {idx.item()}: {deg.item()} connections")

    # Calculate graph center (geometric)
    center = data.pos.mean(dim=0)
    print(f"\nGraph geometric center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")

    # Find nodes closest to center
    distances_to_center = torch.norm(data.pos - center, dim=1)
    closest_nodes = torch.argsort(distances_to_center)[:5]
    print(f"\nNodes closest to center:")
    for i, node_idx in enumerate(closest_nodes):
        dist = distances_to_center[node_idx]
        print(f"  {i+1}. Node {node_idx.item()}: {dist:.2f} Å from center")

    print()


def save_and_modify():
    """Example of modifying and saving a graph."""
    print("\n" + "="*70)
    print("Example 7: Modifying and Saving Graphs")
    print("="*70 + "\n")

    graph_path = "data/processed/graphs/1aju_ARG.pt"
    data = torch.load(graph_path)

    print(f"Original graph: {data.num_nodes} nodes, {data.num_edges} edges")

    # Add a new attribute (e.g., target labels)
    data.y = torch.randn(1, 1536)  # Example ligand embedding
    print(f"Added target embedding: {data.y.shape}")

    # Save modified graph
    output_path = "data/processed/graphs/1aju_ARG_modified.pt"
    torch.save(data, output_path)
    print(f"\nSaved modified graph to: {output_path}")

    # Load and verify
    data_loaded = torch.load(output_path)
    print(f"Loaded modified graph: has target = {hasattr(data_loaded, 'y')}")
    if hasattr(data_loaded, 'y'):
        print(f"Target shape: {data_loaded.y.shape}")

    print()


def main():
    """Run all examples."""
    print("\n" + "🔍 Graph File Reading Examples " + "🔍")

    # Run examples
    basic_read_example()
    access_node_features()
    access_edges()
    calculate_distances()
    batch_loading()
    advanced_analysis()
    save_and_modify()

    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
