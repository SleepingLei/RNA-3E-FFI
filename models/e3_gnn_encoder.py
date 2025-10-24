#!/usr/bin/env python3
"""
E(3) Equivariant Graph Neural Network Encoder

This module implements an E(3) equivariant GNN for encoding RNA binding pockets
using e3nn for spherical harmonics and tensor products.

IMPROVED VERSION with:
- Bessel basis functions (better than Gaussian RBF)
- Polynomial cutoff (smooth distance truncation)
- Gate activation (equivariant nonlinearity)
- Proper residual connections
- Layer normalization
- Average neighbors normalization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter
import e3nn
from e3nn import o3
from e3nn.nn import Gate, NormActivation
from e3nn.o3 import Irreps, Linear, FullyConnectedTensorProduct

# Import improved components
try:
    from .layers import (
        BesselBasis,
        PolynomialCutoff,
        EquivariantLayerNorm
    )
    _has_improved_layers = True
except ImportError:
    _has_improved_layers = False
    import warnings
    warnings.warn(
        "Improved layers not found. Using basic implementation. "
        "Install missing dependencies or check layers/ directory."
    )


class E3GNNMessagePassingLayer(MessagePassing):
    """
    E(3) equivariant message passing layer using e3nn.

    IMPROVED with:
    - Bessel basis instead of Gaussian RBF
    - Polynomial cutoff for smooth truncation
    - Gate activation for nonlinearity
    - Residual connections
    - Average neighbors normalization
    - Optional layer normalization
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_sh="0e + 1o + 2e",
        num_radial_basis=8,
        radial_hidden_dim=64,
        edge_attr_dim=0,
        r_max=6.0,
        avg_num_neighbors=None,
        use_gate=True,
        use_sc=True,
        use_resnet=True,
        use_layer_norm=False
    ):
        """
        Args:
            irreps_in: Input irreps (e3nn irreducible representations)
            irreps_out: Output irreps
            irreps_sh: Spherical harmonics irreps
            num_radial_basis: Number of radial basis functions
            radial_hidden_dim: Hidden dimension for radial MLP
            edge_attr_dim: Dimension of edge attributes (if any)
            r_max: Maximum cutoff distance
            avg_num_neighbors: Average number of neighbors for normalization
            use_gate: Whether to use gate activation
            use_sc: Whether to use self-connection
            use_resnet: Whether to add residual connection
            use_layer_norm: Whether to apply layer normalization
        """
        super().__init__(aggr='add', node_dim=0)

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.avg_num_neighbors = avg_num_neighbors
        self.use_sc = use_sc
        self.use_layer_norm = use_layer_norm

        # Determine if we can use residual connection
        self.use_resnet = use_resnet and (self.irreps_in == self.irreps_out)

        # Build gate activation or use simple version
        if use_gate and _has_improved_layers:
            # Separate scalars and non-scalars for gating
            irreps_scalars = o3.Irreps([
                (mul, ir) for mul, ir in self.irreps_out if ir.l == 0
            ])
            irreps_gated = o3.Irreps([
                (mul, ir) for mul, ir in self.irreps_out if ir.l > 0
            ])

            if irreps_gated.num_irreps > 0:
                irreps_gates = o3.Irreps([
                    (mul, "0e") for mul, _ in irreps_gated
                ])

                self.nonlinearity = Gate(
                    irreps_scalars=irreps_scalars,
                    act_scalars=[torch.nn.functional.silu for _ in irreps_scalars],
                    irreps_gates=irreps_gates,
                    act_gates=[torch.sigmoid for _ in irreps_gates],
                    irreps_gated=irreps_gated
                )
                irreps_before_gate = self.nonlinearity.irreps_in
            else:
                # No gated irreps, use simple activation
                irreps_before_gate = self.irreps_out
                self.nonlinearity = lambda x: x
        else:
            # No gate activation
            irreps_before_gate = self.irreps_out
            self.nonlinearity = lambda x: x

        # Pre-linear layer
        self.linear_in = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_in,
            internal_weights=True,
            shared_weights=True
        )

        # Tensor product for combining node features with spherical harmonics
        self.tp = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            irreps_before_gate,
            shared_weights=False,
            internal_weights=False
        )

        # Radial basis functions
        if _has_improved_layers:
            # Use Bessel basis
            self.bessel_basis = BesselBasis(
                r_max=r_max,
                num_basis=num_radial_basis,
                trainable=False
            )
            self.cutoff = PolynomialCutoff(r_max=r_max, p=6)
        else:
            # Fallback to Gaussian RBF
            self.register_buffer(
                'rbf_centers',
                torch.linspace(0, r_max, num_radial_basis)
            )
            self.register_buffer(
                'rbf_widths',
                torch.ones(num_radial_basis) * 0.5
            )
            self.bessel_basis = None
            self.cutoff = None

        # Radial MLP to generate tensor product weights
        self.radial_mlp = nn.Sequential(
            nn.Linear(num_radial_basis + edge_attr_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, self.tp.weight_numel)
        )

        # Self-connection
        self.self_connection = None
        if self.use_sc:
            self.self_connection = Linear(
                self.irreps_in,
                self.irreps_out,
                internal_weights=True,
                shared_weights=True
            )

        # Layer normalization
        self.layer_norm = None
        if self.use_layer_norm and _has_improved_layers:
            self.layer_norm = EquivariantLayerNorm(self.irreps_out)

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
        # Save for residual
        x_residual = x

        # Self-connection
        if self.self_connection is not None:
            x_self = self.self_connection(x)
        else:
            x_self = 0

        # Apply pre-linear
        x = self.linear_in(x)

        # Message passing
        x_message = self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)

        # Average neighbors normalization
        if self.avg_num_neighbors is not None:
            x_message = x_message / (self.avg_num_neighbors ** 0.5)

        # Combine self-connection and messages
        x = x_self + x_message

        # Apply nonlinearity
        x = self.nonlinearity(x)

        # Residual connection
        if self.use_resnet:
            x = x_residual + x

        # Layer normalization
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x

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

        # Compute radial basis functions
        if self.bessel_basis is not None:
            # Use Bessel basis with cutoff
            rbf = self.bessel_basis(distance)
            cutoff = self.cutoff(distance)
            rbf = rbf * cutoff
        else:
            # Use Gaussian RBF (fallback)
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

    IMPROVED VERSION with all optimizations from e3_layers reference.
    """

    def __init__(
        self,
        input_dim,
        hidden_irreps="32x0e + 16x1o + 8x2e",
        output_dim=512,
        num_layers=4,
        num_radial_basis=8,
        radial_hidden_dim=64,
        pooling_hidden_dim=128,
        r_max=6.0,
        avg_num_neighbors=None,
        use_gate=True,
        use_layer_norm=False,
        pooling_type='attention',
        dropout=0.0
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
            r_max: Maximum cutoff distance (Angstroms)
            avg_num_neighbors: Average number of neighbors for normalization
            use_gate: Whether to use gate activation
            use_layer_norm: Whether to use layer normalization
            pooling_type: Type of pooling ('attention', 'mean', 'sum', 'max')
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.pooling_type = pooling_type
        self.dropout = dropout

        # Initial embedding layer: map input features to irreps
        self.input_irreps = o3.Irreps(f"{input_dim}x0e")

        self.input_embedding = o3.Linear(
            self.input_irreps,
            self.hidden_irreps
        )

        # Message passing layers with improvements
        self.mp_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = E3GNNMessagePassingLayer(
                irreps_in=self.hidden_irreps,
                irreps_out=self.hidden_irreps,
                irreps_sh="0e + 1o + 2e",
                num_radial_basis=num_radial_basis,
                radial_hidden_dim=radial_hidden_dim,
                r_max=r_max,
                avg_num_neighbors=avg_num_neighbors,
                use_gate=use_gate,
                use_sc=True,
                use_resnet=True,
                use_layer_norm=use_layer_norm
            )
            self.mp_layers.append(layer)

        # Extract scalar (invariant) features for pooling
        scalar_irreps = o3.Irreps([(mul, ir) for mul, ir in self.hidden_irreps if ir.l == 0])
        self.scalar_dim = scalar_irreps.dim

        # Pooling mechanism
        if pooling_type == 'attention':
            # Attention-based pooling
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
        for i, layer in enumerate(self.mp_layers):
            h = layer(h, pos, edge_index)

            # Optional dropout between layers (only on scalars)
            if self.dropout > 0 and self.training:
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

        # Extract scalar (invariant) features
        h_scalar = h[:, :self.scalar_dim]

        # Pooling
        if self.pooling_type == 'attention' and self.pooling_mlp is not None:
            # Attention-based weighted pooling
            attention_logits = self.pooling_mlp(h_scalar)
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
            # Default to attention
            attention_logits = self.pooling_mlp(h_scalar) if self.pooling_mlp is not None else h_scalar.mean(dim=-1, keepdim=True)
            attention_weights = softmax(attention_logits, index=batch, dim=0)
            weighted_features = h_scalar * attention_weights
            graph_embedding = scatter(
                weighted_features,
                index=batch,
                dim=0,
                reduce='sum'
            )

        # Project to output dimension
        output = self.output_projection(graph_embedding)

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

    def get_attention_weights(self, data):
        """
        Get attention weights for each node (only for attention pooling).

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Attention weights [num_nodes, 1]
        """
        if self.pooling_type != 'attention' or self.pooling_mlp is None:
            raise ValueError("Attention weights only available for attention pooling")

        h_scalar = self.get_node_embeddings(data)

        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
                else torch.zeros(h_scalar.size(0), dtype=torch.long, device=h_scalar.device)

        attention_logits = self.pooling_mlp(h_scalar)
        attention_weights = softmax(attention_logits, index=batch, dim=0)

        return attention_weights


# Example usage
if __name__ == "__main__":
    # Test the improved model
    from torch_geometric.data import Data, Batch

    print("=" * 80)
    print("Testing IMPROVED E(3) GNN Encoder")
    print("=" * 80)

    # Create dummy data
    num_nodes = 100
    input_dim = 10

    data = Data(
        x=torch.randn(num_nodes, input_dim),
        pos=torch.randn(num_nodes, 3),
        edge_index=torch.randint(0, num_nodes, (2, 300))
    )

    # Create model with improvements
    model = RNAPocketEncoder(
        input_dim=input_dim,
        hidden_irreps="32x0e + 16x1o + 8x2e",
        output_dim=512,
        num_layers=3,
        use_gate=True,
        use_layer_norm=True,
        avg_num_neighbors=10
    )

    # Forward pass
    output = model(data)
    print(f"\nInput shape: {data.x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test with batch
    data_list = [data, data]
    batch = Batch.from_data_list(data_list)
    output_batch = model(batch)
    print(f"Batched output shape: {output_batch.shape}")

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
