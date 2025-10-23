#!/usr/bin/env python3
"""
Test script to verify the inference fix for single graph processing.
"""
import torch
from pathlib import Path
import sys

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent))
from models.e3_gnn_encoder import RNAPocketEncoder
from torch_geometric.data import Data

print("Testing single graph inference fix...")
print("=" * 60)

# Create a simple test graph (without batch attribute)
x = torch.randn(10, 11)  # 10 nodes, 11-dim features
pos = torch.randn(10, 3)  # 10 nodes, 3D coordinates
edge_index = torch.randint(0, 10, (2, 20))  # 20 edges

data = Data(x=x, pos=pos, edge_index=edge_index)
print(f"✓ Created test graph: {data.num_nodes} nodes, {data.num_edges} edges")
print(f"  Has batch attribute: {hasattr(data, 'batch')}")

# Create model
print("\nInitializing model...")
model = RNAPocketEncoder(
    input_dim=11,
    hidden_irreps='32x0e + 16x1o + 8x2e',
    output_dim=1536,
    num_layers=4,
    num_radial_basis=8
)
print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

# Test inference
print("\nRunning inference...")
model.eval()
with torch.no_grad():
    output = model(data)
    print(f"✓ Inference successful!")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected shape: (1, 1536)")

    if output.shape == (1, 1536):
        print("\n✅ TEST PASSED: Single graph inference works correctly!")
    else:
        print(f"\n❌ TEST FAILED: Expected shape (1, 1536), got {output.shape}")
        sys.exit(1)

# Test with batch
print("\n" + "=" * 60)
print("Testing batched graph inference...")
from torch_geometric.loader import DataLoader

# Create multiple graphs
graphs = [
    Data(x=torch.randn(10, 11), pos=torch.randn(10, 3), edge_index=torch.randint(0, 10, (2, 20)))
    for _ in range(3)
]

loader = DataLoader(graphs, batch_size=3, shuffle=False)
batch_data = next(iter(loader))

print(f"✓ Created batched data: {batch_data.num_graphs} graphs")
print(f"  Total nodes: {batch_data.num_nodes}")
print(f"  Has batch attribute: {hasattr(batch_data, 'batch')}")

with torch.no_grad():
    output = model(batch_data)
    print(f"✓ Batched inference successful!")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected shape: (3, 1536)")

    if output.shape == (3, 1536):
        print("\n✅ TEST PASSED: Batched graph inference works correctly!")
    else:
        print(f"\n❌ TEST FAILED: Expected shape (3, 1536), got {output.shape}")
        sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED! The inference fix is working correctly.")
print("\nYou can now run the inference script on the remote server:")
print("python scripts/05_run_inference.py \\")
print("  --checkpoint models/checkpoints/best_model.pt \\")
print("  --query_graph data/processed/graphs/1aju_ARG_model0.pt \\")
print("  --ligand_library data/processed/ligand_embeddings.h5 \\")
print("  --top_k 10 \\")
print("  --output results/predictions.json")
