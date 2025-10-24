#!/usr/bin/env python3
"""
Improved E(3) Equivariant Graph Neural Network Encoder

This module implements an enhanced E(3) equivariant GNN for encoding RNA binding pockets
with all optimizations from e3_layers reference implementation:

Key improvements:
1. Bessel basis functions instead of Gaussian RBF
2. Polynomial cutoff for smooth distance truncation
3. Gate activation for maintaining equivariance with nonlinearity
4. Proper residual connections
5. Layer normalization
6. Average neighbors normalization
7. Improved pooling mechanisms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter
from e3nn import o3
from e3nn.o3 import Irreps, Linear

from .layers import (
    BesselBasis,
    PolynomialCutoff,
    ImprovedE3MessagePassingLayer,
    EquivariantLayerNorm
)


class ImprovedRNAPocketEncoder(nn.Module):
    """
    Enhanced E(3) Equivariant GNN for encoding RNA binding pockets.

    This model incorporates best practices from the e3_layers reference:
    - Bessel basis for better physical modeling
    - Smooth polynomial cutoff
    - Gate activation for equivariant nonlinearity
    - Residual connections for deeper networks
    - Layer normalization for training stability
    - Average neighbors normalization for handling varying node degrees

    Args:
        input_dim: Dimension of input node features
        hidden_irreps: Hidden layer irreps (e.g., "32x0e + 16x1o + 8x2e")
        output_dim: Dimension of final pocket embedding
        num_layers: Number of message passing layers
        r_max: Maximum cutoff distance (Angstroms)
        num_radial_basis: Number of Bessel basis functions
        radial_hidden_dim: Hidden dimension for radial MLPs
        pooling_type: Type of pooling ('attention', 'mean', 'sum', 'max')
        pooling_hidden_dim: Hidden dimension for attention pooling
        avg_num_neighbors: Average number of neighbors for normalization
        use_gate: Whether to use gate activation
        use_layer_norm: Whether to use layer normalization
        dropout: Dropout rate (0 to disable)
    """

    def __init__(
        self,
        input_dim,
        hidden_irreps="32x0e + 16x1o + 8x2e",
        output_dim=512,
        num_layers=4,
        r_max=6.0,
        num_radial_basis=8,
        radial_hidden_dim=64,
        pooling_type='attention',
        pooling_hidden_dim=128,
        avg_num_neighbors=None,
        use_gate=True,
        use_layer_norm=False,
        dropout=0.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.pooling_type = pooling_type
        self.dropout = dropout

        # Input embedding
        self.input_irreps = o3.Irreps(f"{input_dim}x0e")
        self.input_embedding = Linear(
            self.input_irreps,
            self.hidden_irreps,
            internal_weights=True,
            shared_weights=True
        )

        # Message passing layers with all improvements
        self.mp_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = ImprovedE3MessagePassingLayer(
                irreps_in=self.hidden_irreps,
                irreps_out=self.hidden_irreps,
                irreps_sh="0e + 1o + 2e",
                r_max=r_max,
                num_radial_basis=num_radial_basis,
                radial_hidden_dim=radial_hidden_dim,
                avg_num_neighbors=avg_num_neighbors,
                use_gate=use_gate,
                use_sc=True,
                use_resnet=True,
                use_layer_norm=use_layer_norm
            )
            self.mp_layers.append(layer)

        # Extract scalar (invariant) features for pooling
        scalar_irreps = o3.Irreps([
            (mul, ir) for mul, ir in self.hidden_irreps if ir.l == 0
        ])
        self.scalar_dim = scalar_irreps.dim

        # Pooling mechanism
        if pooling_type == 'attention':
            # Learned attention-based pooling
            self.pooling_mlp = nn.Sequential(
                nn.Linear(self.scalar_dim, pooling_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(pooling_hidden_dim, pooling_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(pooling_hidden_dim, 1)
            )
        else:
            self.pooling_mlp = None

        # Output projection with better architecture
        self.output_projection = nn.Sequential(
            nn.Linear(self.scalar_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, data):
        """
        Forward pass through the improved model.

        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features [num_nodes, input_dim]
                - pos: Node positions [num_nodes, 3]
                - edge_index: Edge indices [2, num_edges]
                - batch: Batch indices [num_nodes] (optional)

        Returns:
            Pocket embeddings [batch_size, output_dim]
        """
        x, pos, edge_index = data.x, data.pos, data.edge_index

        # Get batch information
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Initial embedding
        h = self.input_embedding(x)

        # Message passing with all improvements
        for i, layer in enumerate(self.mp_layers):
            h = layer(h, pos, edge_index)

            # Optional dropout between layers
            if self.dropout > 0 and self.training:
                # Only apply to scalar features to maintain equivariance
                scalar_mask = torch.zeros(h.size(-1), dtype=torch.bool, device=h.device)
                idx = 0
                for mul, ir in self.hidden_irreps:
                    if ir.l == 0:
                        scalar_mask[idx:idx + mul * ir.dim] = True
                    idx += mul * ir.dim

                h_dropped = h.clone()
                h_dropped[..., scalar_mask] = F.dropout(
                    h[..., scalar_mask],
                    p=self.dropout,
                    training=self.training
                )
                h = h_dropped

        # Extract scalar features
        h_scalar = h[:, :self.scalar_dim]

        # Pooling
        if self.pooling_type == 'attention':
            # Attention-based weighted pooling
            attention_logits = self.pooling_mlp(h_scalar)  # [num_nodes, 1]
            attention_weights = softmax(attention_logits, index=batch, dim=0)
            weighted_features = h_scalar * attention_weights
            graph_embedding = scatter(
                weighted_features,
                index=batch,
                dim=0,
                reduce='sum'
            )
        elif self.pooling_type == 'mean':
            graph_embedding = scatter(h_scalar, index=batch, dim=0, reduce='mean')
        elif self.pooling_type == 'sum':
            graph_embedding = scatter(h_scalar, index=batch, dim=0, reduce='sum')
        elif self.pooling_type == 'max':
            graph_embedding = scatter(h_scalar, index=batch, dim=0, reduce='max')
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # Project to output dimension
        output = self.output_projection(graph_embedding)

        return output

    def get_node_embeddings(self, data):
        """
        Get per-node embeddings before pooling.

        Useful for visualization, interpretability, and downstream tasks.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Node embeddings [num_nodes, scalar_dim]
        """
        x, pos, edge_index = data.x, data.pos, data.edge_index

        # Initial embedding
        h = self.input_embedding(x)

        # Message passing
        for layer in self.mp_layers:
            h = layer(h, pos, edge_index)

        # Extract scalar features
        h_scalar = h[:, :self.scalar_dim]

        return h_scalar

    def get_attention_weights(self, data):
        """
        Get attention weights for each node (only for attention pooling).

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Attention weights [num_nodes, 1]
        """
        if self.pooling_type != 'attention':
            raise ValueError("Attention weights only available for attention pooling")

        h_scalar = self.get_node_embeddings(data)

        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
                else torch.zeros(h_scalar.size(0), dtype=torch.long, device=h_scalar.device)

        attention_logits = self.pooling_mlp(h_scalar)
        attention_weights = softmax(attention_logits, index=batch, dim=0)

        return attention_weights


class HierarchicalRNAPocketEncoder(nn.Module):
    """
    Hierarchical encoder with multi-scale message passing.

    Processes information at multiple scales using different cutoff distances.

    Args:
        input_dim: Input feature dimension
        hidden_irreps: Hidden irreps
        output_dim: Output dimension
        num_layers_per_scale: Layers per scale
        scales: List of cutoff distances for each scale
        num_radial_basis: Number of basis functions
        pooling_type: Pooling method
    """

    def __init__(
        self,
        input_dim,
        hidden_irreps="32x0e + 16x1o + 8x2e",
        output_dim=512,
        num_layers_per_scale=2,
        scales=[3.0, 6.0, 10.0],
        num_radial_basis=8,
        pooling_type='attention'
    ):
        super().__init__()

        self.scales = scales
        self.num_scales = len(scales)

        # Input embedding
        input_irreps = o3.Irreps(f"{input_dim}x0e")
        hidden_irreps = o3.Irreps(hidden_irreps)

        self.input_embedding = Linear(input_irreps, hidden_irreps)

        # Multi-scale encoders
        self.scale_encoders = nn.ModuleList()
        for r_max in scales:
            encoder_layers = nn.ModuleList()
            for _ in range(num_layers_per_scale):
                layer = ImprovedE3MessagePassingLayer(
                    irreps_in=hidden_irreps,
                    irreps_out=hidden_irreps,
                    r_max=r_max,
                    num_radial_basis=num_radial_basis,
                    use_gate=True,
                    use_resnet=True
                )
                encoder_layers.append(layer)
            self.scale_encoders.append(encoder_layers)

        # Get scalar dimension
        scalar_irreps = o3.Irreps([
            (mul, ir) for mul, ir in hidden_irreps if ir.l == 0
        ])
        scalar_dim = scalar_irreps.dim

        # Pooling for each scale
        self.pooling_mlps = nn.ModuleList()
        for _ in scales:
            if pooling_type == 'attention':
                mlp = nn.Sequential(
                    nn.Linear(scalar_dim, 64),
                    nn.SiLU(),
                    nn.Linear(64, 1)
                )
            else:
                mlp = None
            self.pooling_mlps.append(mlp)

        # Combine multi-scale features
        self.fusion = nn.Sequential(
            nn.Linear(scalar_dim * self.num_scales, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.SiLU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

        self.pooling_type = pooling_type
        self.scalar_dim = scalar_dim

    def forward(self, data):
        """Forward pass through hierarchical encoder."""
        x, pos, edge_index = data.x, data.pos, data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
                else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Initial embedding
        h = self.input_embedding(x)

        # Process at each scale
        scale_embeddings = []
        for scale_layers, pooling_mlp in zip(self.scale_encoders, self.pooling_mlps):
            h_scale = h

            # Message passing at this scale
            for layer in scale_layers:
                h_scale = layer(h_scale, pos, edge_index)

            # Extract scalars and pool
            h_scalar = h_scale[:, :self.scalar_dim]

            if self.pooling_type == 'attention' and pooling_mlp is not None:
                attention_logits = pooling_mlp(h_scalar)
                attention_weights = softmax(attention_logits, index=batch, dim=0)
                weighted = h_scalar * attention_weights
                pooled = scatter(weighted, index=batch, dim=0, reduce='sum')
            else:
                pooled = scatter(h_scalar, index=batch, dim=0, reduce='mean')

            scale_embeddings.append(pooled)

        # Concatenate multi-scale features
        combined = torch.cat(scale_embeddings, dim=-1)

        # Fusion
        output = self.fusion(combined)

        return output


# Example usage and testing
if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    print("=" * 60)
    print("Testing Improved RNA Pocket Encoder")
    print("=" * 60)

    # Create dummy data
    num_nodes = 100
    input_dim = 10

    data = Data(
        x=torch.randn(num_nodes, input_dim),
        pos=torch.randn(num_nodes, 3) * 5,  # Random positions in space
        edge_index=torch.randint(0, num_nodes, (2, 300))
    )

    # Test basic model
    print("\n1. Testing Basic Improved Model")
    model = ImprovedRNAPocketEncoder(
        input_dim=input_dim,
        hidden_irreps="32x0e + 16x1o + 8x2e",
        output_dim=512,
        num_layers=3,
        r_max=6.0,
        num_radial_basis=8,
        use_gate=True,
        use_layer_norm=True,
        pooling_type='attention'
    )

    output = model(data)
    print(f"   Input shape: {data.x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with batch
    print("\n2. Testing Batched Input")
    data_list = [data, data, data]
    batch_data = Batch.from_data_list(data_list)
    output_batch = model(batch_data)
    print(f"   Batched output shape: {output_batch.shape}")

    # Test node embeddings
    print("\n3. Testing Node Embeddings")
    node_emb = model.get_node_embeddings(data)
    print(f"   Node embeddings shape: {node_emb.shape}")

    # Test attention weights
    print("\n4. Testing Attention Weights")
    attention = model.get_attention_weights(data)
    print(f"   Attention weights shape: {attention.shape}")
    print(f"   Attention sum: {attention.sum():.4f}")

    # Test hierarchical model
    print("\n5. Testing Hierarchical Model")
    hier_model = HierarchicalRNAPocketEncoder(
        input_dim=input_dim,
        hidden_irreps="16x0e + 8x1o + 4x2e",
        output_dim=512,
        num_layers_per_scale=2,
        scales=[3.0, 6.0, 10.0]
    )

    hier_output = hier_model(data)
    print(f"   Hierarchical output shape: {hier_output.shape}")
    print(f"   Hierarchical model parameters: {sum(p.numel() for p in hier_model.parameters()):,}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
