#!/usr/bin/env python3
"""
Script to inspect and visualize PyTorch Geometric graph files.

This script loads a .pt graph file and displays its structure and content.
"""
import sys
import argparse
from pathlib import Path
import torch
import numpy as np


def inspect_graph(graph_path):
    """
    Load and inspect a graph file.

    Args:
        graph_path: Path to the .pt graph file
    """
    print(f"\n{'='*70}")
    print(f"Inspecting graph: {graph_path}")
    print(f"{'='*70}\n")

    # Load the graph
    data = torch.load(graph_path)

    # Basic information
    print("📊 Graph Overview:")
    print("-" * 70)
    print(f"Type: {type(data)}")
    print(f"Number of nodes: {data.num_nodes}")
    if hasattr(data, 'num_edges'):
        print(f"Number of edges: {data.num_edges}")
    print()

    # Node features
    if hasattr(data, 'x') and data.x is not None:
        print("🔢 Node Features (x):")
        print("-" * 70)
        print(f"Shape: {data.x.shape}")
        print(f"Data type: {data.x.dtype}")
        print(f"Value range: [{data.x.min():.4f}, {data.x.max():.4f}]")
        print(f"Mean: {data.x.mean():.4f}, Std: {data.x.std():.4f}")
        print(f"\nFirst 3 nodes features:")
        print(data.x[:3])
        print()

    # Node positions
    if hasattr(data, 'pos') and data.pos is not None:
        print("📍 Node Positions (pos):")
        print("-" * 70)
        print(f"Shape: {data.pos.shape}")
        print(f"Data type: {data.pos.dtype}")
        print(f"X range: [{data.pos[:, 0].min():.2f}, {data.pos[:, 0].max():.2f}]")
        print(f"Y range: [{data.pos[:, 1].min():.2f}, {data.pos[:, 1].max():.2f}]")
        print(f"Z range: [{data.pos[:, 2].min():.2f}, {data.pos[:, 2].max():.2f}]")
        print(f"\nFirst 3 node positions:")
        print(data.pos[:3])
        print()

    # Edge indices
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        print("🔗 Edge Indices (edge_index):")
        print("-" * 70)
        print(f"Shape: {data.edge_index.shape}")
        print(f"Data type: {data.edge_index.dtype}")
        print(f"Number of edges: {data.edge_index.shape[1]}")
        print(f"\nFirst 5 edges (source -> target):")
        for i in range(min(5, data.edge_index.shape[1])):
            src = data.edge_index[0, i].item()
            dst = data.edge_index[1, i].item()
            print(f"  Edge {i}: {src} -> {dst}")
        print()

    # Edge attributes
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        print("📏 Edge Attributes (edge_attr):")
        print("-" * 70)
        print(f"Shape: {data.edge_attr.shape}")
        print(f"Data type: {data.edge_attr.dtype}")
        print(f"Value range: [{data.edge_attr.min():.4f}, {data.edge_attr.max():.4f}]")
        print(f"\nFirst 3 edge attributes:")
        print(data.edge_attr[:3])
        print()

    # Batch information
    if hasattr(data, 'batch') and data.batch is not None:
        print("📦 Batch Information:")
        print("-" * 70)
        print(f"Shape: {data.batch.shape}")
        print(f"Number of graphs in batch: {data.batch.max().item() + 1}")
        print()

    # Target/Label
    if hasattr(data, 'y') and data.y is not None:
        print("🎯 Target/Label (y):")
        print("-" * 70)
        print(f"Shape: {data.y.shape}")
        print(f"Data type: {data.y.dtype}")
        if data.y.numel() <= 10:
            print(f"Values: {data.y}")
        else:
            print(f"Value range: [{data.y.min():.4f}, {data.y.max():.4f}]")
            print(f"First 5 values: {data.y[:5]}")
        print()

    # Additional attributes
    print("📝 Additional Attributes:")
    print("-" * 70)
    all_attrs = dir(data)
    standard_attrs = {'x', 'pos', 'edge_index', 'edge_attr', 'batch', 'y', 'num_nodes', 'num_edges'}
    custom_attrs = [attr for attr in all_attrs if not attr.startswith('_') and attr not in standard_attrs]

    if custom_attrs:
        for attr in custom_attrs:
            try:
                value = getattr(data, attr)
                if not callable(value):
                    print(f"  {attr}: {value}")
            except:
                pass
    else:
        print("  None")
    print()

    # Memory usage
    print("💾 Memory Usage:")
    print("-" * 70)
    total_bytes = 0
    for attr in ['x', 'pos', 'edge_index', 'edge_attr', 'batch', 'y']:
        if hasattr(data, attr):
            tensor = getattr(data, attr)
            if tensor is not None and isinstance(tensor, torch.Tensor):
                bytes_used = tensor.element_size() * tensor.nelement()
                total_bytes += bytes_used
                print(f"  {attr}: {bytes_used / 1024:.2f} KB")
    print(f"  Total: {total_bytes / 1024:.2f} KB ({total_bytes / (1024*1024):.2f} MB)")
    print()

    # Summary statistics
    print("📈 Summary:")
    print("-" * 70)
    if hasattr(data, 'x') and hasattr(data, 'edge_index'):
        avg_degree = data.edge_index.shape[1] / data.num_nodes
        print(f"Average node degree: {avg_degree:.2f}")

    if hasattr(data, 'pos'):
        # Calculate graph extent
        extent = data.pos.max(dim=0)[0] - data.pos.min(dim=0)[0]
        print(f"Graph spatial extent: [{extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f}] Å")

        # Calculate center of mass
        center = data.pos.mean(dim=0)
        print(f"Center of mass: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")

    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect PyTorch Geometric graph files")
    parser.add_argument("graph_file", type=str, help="Path to the .pt graph file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show more details")

    args = parser.parse_args()

    graph_path = Path(args.graph_file)

    if not graph_path.exists():
        print(f"Error: File {graph_path} does not exist!")
        sys.exit(1)

    inspect_graph(graph_path)


if __name__ == "__main__":
    main()
