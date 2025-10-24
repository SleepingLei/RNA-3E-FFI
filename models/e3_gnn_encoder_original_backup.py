#!/usr/bin/env python3
"""
E(3) Equivariant Graph Neural Network Encoder

This module implements an E(3) equivariant GNN for encoding RNA binding pockets
using e3nn for spherical harmonics and tensor products.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter
import e3nn
from e3nn import o3
from e3nn.nn import Gate


class E3GNNMessagePassingLayer(MessagePassing):
    """
    E(3) equivariant message passing layer using e3nn.

    This layer performs message passing while maintaining E(3) equivariance
    through the use of spherical harmonics and tensor products.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_sh="0e + 1o + 2e",
        num_radial_basis=8,
        radial_hidden_dim=64,
        edge_attr_dim=0
    ):
        """
        Args:
            irreps_in: Input irreps (e3nn irreducible representations)
            irreps_out: Output irreps
            irreps_sh: Spherical harmonics irreps
            num_radial_basis: Number of radial basis functions
            radial_hidden_dim: Hidden dimension for radial MLP
            edge_attr_dim: Dimension of edge attributes (if any)
        """
        super().__init__(aggr='add', node_dim=0)

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)

        # Tensor product for combining node features with spherical harmonics
        # Use FullyConnectedTensorProduct for compatibility with newer e3nn versions
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            shared_weights=False,
            internal_weights=False
        )

        # Radial MLP to generate tensor product weights
        # Maps distance -> weights for tensor product
        self.radial_mlp = nn.Sequential(
            nn.Linear(num_radial_basis + edge_attr_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, self.tp.weight_numel)
        )

        # Radial basis parameters (Gaussian RBF)
        self.num_radial_basis = num_radial_basis
        self.register_buffer(
            'rbf_centers',
            torch.linspace(0, 6, num_radial_basis)  # 0-6 Angstroms
        )
        self.register_buffer(
            'rbf_widths',
            torch.ones(num_radial_basis) * 0.5
        )

        # Self-interaction (skip connection)
        self.self_interaction = o3.Linear(self.irreps_in, self.irreps_out)

    def forward(self, x, pos, edge_index, edge_attr=None):
        """
        Forward pass of message passing layer.

        Args:
            x: Node features [num_nodes, irreps_in_dim]
            pos: Node positions [num_nodes, 3]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Optional edge attributes [num_edges, edge_attr_dim]

        Returns:
            Updated node features [num_nodes, irreps_out_dim]
        """
        # Self-interaction term
        x_self = self.self_interaction(x)

        # Message passing
        x_message = self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)

        # Combine self-interaction and messages
        return x_self + x_message

    def message(self, x_j, pos_i, pos_j, edge_attr=None):
        """
        Construct messages from source nodes to target nodes.

        Args:
            x_j: Features of source nodes [num_edges, irreps_in_dim]
            pos_i: Positions of target nodes [num_edges, 3]
            pos_j: Positions of source nodes [num_edges, 3]
            edge_attr: Optional edge attributes [num_edges, edge_attr_dim]

        Returns:
            Messages [num_edges, irreps_out_dim]
        """
        # Compute relative positions
        rel_pos = pos_i - pos_j  # [num_edges, 3]
        distance = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)  # [num_edges, 1]

        # Avoid division by zero
        distance = torch.clamp(distance, min=1e-6)

        # Normalize relative positions
        rel_pos_normalized = rel_pos / distance

        # Compute spherical harmonics
        sh = o3.spherical_harmonics(
            self.irreps_sh,
            rel_pos_normalized,
            normalize=True,
            normalization='component'
        )

        # Compute radial basis functions (Gaussian RBF)
        rbf = torch.exp(-((distance - self.rbf_centers) ** 2) / (2 * self.rbf_widths ** 2))
        rbf = rbf / (torch.sqrt(2 * torch.tensor(3.14159)) * self.rbf_widths)

        # Concatenate RBF with edge attributes if present
        if edge_attr is not None:
            radial_input = torch.cat([rbf, edge_attr], dim=-1)
        else:
            radial_input = rbf

        # Generate tensor product weights from radial features
        tp_weights = self.radial_mlp(radial_input)

        # Apply tensor product: combine node features with spherical harmonics
        messages = self.tp(x_j, sh, tp_weights)

        return messages


class RNAPocketEncoder(nn.Module):
    """
    E(3) Equivariant GNN for encoding RNA binding pockets.

    This model takes a molecular graph with 3D positions and produces
    an invariant embedding vector through equivariant message passing
    and weighted pooling.
    """

    def __init__(
        self,
        input_dim,
        hidden_irreps="32x0e + 16x1o + 8x2e",
        output_dim=512,
        num_layers=4,
        num_radial_basis=8,
        radial_hidden_dim=64,
        pooling_hidden_dim=128
    ):
        """
        Args:
            input_dim: Dimension of input node features
            hidden_irreps: Hidden layer irreps specification
            output_dim: Dimension of final pocket embedding
            num_layers: Number of message passing layers
            num_radial_basis: Number of radial basis functions
            radial_hidden_dim: Hidden dimension for radial MLPs
            pooling_hidden_dim: Hidden dimension for pooling MLP
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Initial embedding layer: map input features to irreps
        # For simplicity, we'll use only scalar features (0e) for input
        self.input_irreps = o3.Irreps(f"{input_dim}x0e")

        self.input_embedding = o3.Linear(
            self.input_irreps,
            self.hidden_irreps
        )

        # Message passing layers
        self.mp_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = E3GNNMessagePassingLayer(
                irreps_in=self.hidden_irreps,
                irreps_out=self.hidden_irreps,
                irreps_sh="0e + 1o + 2e",
                num_radial_basis=num_radial_basis,
                radial_hidden_dim=radial_hidden_dim
            )
            self.mp_layers.append(layer)

        # Extract scalar (invariant) features for pooling
        # Only keep 0e (scalar) irreps
        scalar_irreps = o3.Irreps([(mul, ir) for mul, ir in self.hidden_irreps if ir.l == 0])
        self.scalar_dim = scalar_irreps.dim

        # Attention-based pooling
        self.pooling_mlp = nn.Sequential(
            nn.Linear(self.scalar_dim, pooling_hidden_dim),
            nn.SiLU(),
            nn.Linear(pooling_hidden_dim, pooling_hidden_dim),
            nn.SiLU(),
            nn.Linear(pooling_hidden_dim, 1)
        )

        # Final projection to output dimension
        self.output_projection = nn.Sequential(
            nn.Linear(self.scalar_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, data):
        """
        Forward pass through the model.

        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features [num_nodes, input_dim]
                - pos: Node positions [num_nodes, 3]
                - edge_index: Edge indices [2, num_edges]
                - batch: Batch indices [num_nodes] (for batched graphs)

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

        # Message passing
        for layer in self.mp_layers:
            h = layer(h, pos, edge_index)

        # Extract scalar (invariant) features
        # The scalar features are the first scalar_dim elements of h
        h_scalar = h[:, :self.scalar_dim]

        # Parameterized weighted pooling
        # Compute attention weights for each node
        attention_logits = self.pooling_mlp(h_scalar)  # [num_nodes, 1]

        # Apply softmax per graph in batch
        # Use index parameter for PyG softmax
        attention_weights = softmax(attention_logits, index=batch, dim=0)  # [num_nodes, 1]

        # Weighted sum of node features
        weighted_features = h_scalar * attention_weights  # [num_nodes, scalar_dim]

        # Sum over nodes in each graph
        # Use index parameter for PyG scatter
        graph_embedding = scatter(
            weighted_features,
            index=batch,
            dim=0,
            reduce='sum'
        )  # [batch_size, scalar_dim]

        # Project to output dimension
        output = self.output_projection(graph_embedding)  # [batch_size, output_dim]

        return output

    def get_node_embeddings(self, data):
        """
        Get per-node embeddings (before pooling).

        Useful for visualization and analysis.

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


# Example usage
if __name__ == "__main__":
    # Test the model
    from torch_geometric.data import Data, Batch

    # Create dummy data
    num_nodes = 100
    input_dim = 10

    data = Data(
        x=torch.randn(num_nodes, input_dim),
        pos=torch.randn(num_nodes, 3),
        edge_index=torch.randint(0, num_nodes, (2, 300))
    )

    # Create model
    model = RNAPocketEncoder(
        input_dim=input_dim,
        hidden_irreps="32x0e + 16x1o + 8x2e",
        output_dim=512,
        num_layers=3
    )

    # Forward pass
    output = model(data)
    print(f"Input shape: {data.x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Test with batch
    data_list = [data, data]
    batch = Batch.from_data_list(data_list)
    output_batch = model(batch)
    print(f"Batched output shape: {output_batch.shape}")
