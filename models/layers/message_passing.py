#!/usr/bin/env python3
"""
Improved E(3) Equivariant Message Passing Layer

Implements an enhanced message passing layer with:
- Gate activation for maintaining equivariance
- Residual connections
- Average neighbors normalization
- Layer normalization
- Bessel basis and polynomial cutoff
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from e3nn import o3
from e3nn.nn import Gate, NormActivation
from e3nn.o3 import Irreps, Linear, FullyConnectedTensorProduct

from .radial_basis import BesselBasis, PolynomialCutoff
from .normalization import EquivariantLayerNorm


class ImprovedE3MessagePassingLayer(MessagePassing):
    """
    Enhanced E(3) equivariant message passing layer.

    Improvements over basic implementation:
    1. Bessel basis instead of Gaussian RBF
    2. Polynomial cutoff function
    3. Gate activation for nonlinearity
    4. Residual connections (when dimensions match)
    5. Average neighbors normalization
    6. Optional layer normalization

    Args:
        irreps_in: Input irreps
        irreps_out: Output irreps
        irreps_sh: Spherical harmonics irreps
        r_max: Maximum cutoff distance
        num_radial_basis: Number of Bessel basis functions
        radial_hidden_dim: Hidden dimension for radial MLP
        avg_num_neighbors: Average number of neighbors for normalization
        use_gate: Whether to use gate activation (vs NormActivation)
        use_sc: Whether to use self-connection
        use_resnet: Whether to add residual connection
        use_layer_norm: Whether to apply layer normalization
        edge_attr_dim: Dimension of additional edge attributes
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_sh="0e + 1o + 2e",
        r_max=6.0,
        num_radial_basis=8,
        radial_hidden_dim=64,
        avg_num_neighbors=None,
        use_gate=True,
        use_sc=True,
        use_resnet=True,
        use_layer_norm=False,
        edge_attr_dim=0
    ):
        super().__init__(aggr='add', node_dim=0)

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.avg_num_neighbors = avg_num_neighbors
        self.use_sc = use_sc
        self.use_layer_norm = use_layer_norm

        # Determine if we can use residual connection
        self.use_resnet = use_resnet and (self.irreps_in == self.irreps_out)

        # Build gate activation
        # Separate scalars and non-scalars for gating
        irreps_scalars = o3.Irreps([
            (mul, ir) for mul, ir in self.irreps_out if ir.l == 0
        ])
        irreps_gated = o3.Irreps([
            (mul, ir) for mul, ir in self.irreps_out if ir.l > 0
        ])

        # If using gate, we need gate scalars for each gated irrep
        if use_gate and irreps_gated.num_irreps > 0:
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

            # Tensor product outputs to the input of the gate
            irreps_before_gate = self.nonlinearity.irreps_in
        else:
            # Use norm activation instead
            irreps_before_gate = self.irreps_out
            self.nonlinearity = NormActivation(
                irreps_in=irreps_before_gate,
                scalar_nonlinearity=torch.nn.functional.silu,
                normalize=True,
                epsilon=1e-8,
                bias=False
            )

        # Pre-linear layer
        self.linear_in = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_in,
            internal_weights=True,
            shared_weights=True
        )

        # Tensor product for message construction
        self.tp = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            irreps_before_gate,
            shared_weights=False,
            internal_weights=False
        )

        # Radial network: distance -> tensor product weights
        # Use Bessel basis instead of Gaussian
        self.bessel_basis = BesselBasis(
            r_max=r_max,
            num_basis=num_radial_basis,
            trainable=False
        )

        self.cutoff = PolynomialCutoff(r_max=r_max, p=6)

        # Radial MLP
        self.radial_mlp = nn.Sequential(
            nn.Linear(num_radial_basis + edge_attr_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, self.tp.weight_numel)
        )

        # Self-connection (combines node features with node attributes)
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
        if self.use_layer_norm:
            self.layer_norm = EquivariantLayerNorm(self.irreps_out)

    def forward(self, x, pos, edge_index, edge_attr=None):
        """
        Forward pass.

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

        # Apply pre-linear layer
        x = self.linear_in(x)

        # Message passing
        x_message = self.propagate(
            edge_index,
            x=x,
            pos=pos,
            edge_attr=edge_attr
        )

        # Average neighbors normalization
        if self.avg_num_neighbors is not None:
            x_message = x_message / (self.avg_num_neighbors ** 0.5)

        # Apply nonlinearity (gate or norm activation) BEFORE combining
        # This reduces dimension from irreps_before_gate to irreps_out
        x_message = self.nonlinearity(x_message)

        # Combine (now both are irreps_out dimension)
        x = x_self + x_message

        # Residual connection
        if self.use_resnet:
            x = x_residual + x

        # Layer normalization
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x

    def message(self, x_j, pos_i, pos_j, edge_attr=None):
        """
        Construct messages from source to target nodes.

        Args:
            x_j: Source node features [num_edges, irreps_in_dim]
            pos_i: Target positions [num_edges, 3]
            pos_j: Source positions [num_edges, 3]
            edge_attr: Optional edge attributes [num_edges, edge_attr_dim]

        Returns:
            Messages [num_edges, irreps_out_dim]
        """
        # Relative position and distance
        rel_pos = pos_i - pos_j  # [num_edges, 3]
        distance = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)  # [num_edges, 1]

        # Avoid division by zero
        distance = torch.clamp(distance, min=1e-6)
        rel_pos_normalized = rel_pos / distance

        # Spherical harmonics
        sh = o3.spherical_harmonics(
            self.irreps_sh,
            rel_pos_normalized,
            normalize=True,
            normalization='component'
        )  # [num_edges, irreps_sh_dim]

        # Bessel basis with cutoff
        rbf = self.bessel_basis(distance)  # [num_edges, num_basis]
        cutoff = self.cutoff(distance)  # [num_edges, 1]
        rbf = rbf * cutoff  # Apply smooth cutoff

        # Concatenate with edge attributes if present
        if edge_attr is not None:
            radial_features = torch.cat([rbf, edge_attr], dim=-1)
        else:
            radial_features = rbf

        # Generate tensor product weights
        tp_weights = self.radial_mlp(radial_features)

        # Tensor product: combine node features with spherical harmonics
        messages = self.tp(x_j, sh, tp_weights)

        return messages


class MultiLayerE3GNN(nn.Module):
    """
    Stack of improved E(3) message passing layers.

    Args:
        irreps_in: Input irreps
        irreps_hidden: Hidden irreps
        irreps_out: Output irreps
        num_layers: Number of message passing layers
        r_max: Maximum cutoff distance
        num_radial_basis: Number of Bessel basis functions
        radial_hidden_dim: Hidden dimension for radial MLPs
        avg_num_neighbors: Average number of neighbors
        use_gate: Whether to use gate activation
        use_layer_norm: Whether to use layer normalization
    """

    def __init__(
        self,
        irreps_in,
        irreps_hidden,
        irreps_out,
        num_layers=4,
        r_max=6.0,
        num_radial_basis=8,
        radial_hidden_dim=64,
        avg_num_neighbors=None,
        use_gate=True,
        use_layer_norm=False
    ):
        super().__init__()

        self.num_layers = num_layers

        # Input embedding
        self.embed = Linear(irreps_in, irreps_hidden)

        # Message passing layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = ImprovedE3MessagePassingLayer(
                irreps_in=irreps_hidden,
                irreps_out=irreps_hidden,
                r_max=r_max,
                num_radial_basis=num_radial_basis,
                radial_hidden_dim=radial_hidden_dim,
                avg_num_neighbors=avg_num_neighbors,
                use_gate=use_gate,
                use_resnet=True,
                use_layer_norm=use_layer_norm
            )
            self.layers.append(layer)

        # Output projection
        self.project = Linear(irreps_hidden, irreps_out)

    def forward(self, x, pos, edge_index):
        """
        Forward pass through all layers.

        Args:
            x: Node features
            pos: Node positions
            edge_index: Edge indices

        Returns:
            Output features
        """
        # Embed
        x = self.embed(x)

        # Message passing
        for layer in self.layers:
            x = layer(x, pos, edge_index)

        # Project
        x = self.project(x)

        return x


# Example usage
if __name__ == "__main__":
    from torch_geometric.data import Data

    # Create test data
    num_nodes = 50
    num_edges = 200

    data = Data(
        x=torch.randn(num_nodes, 10),
        pos=torch.randn(num_nodes, 3),
        edge_index=torch.randint(0, num_nodes, (2, num_edges))
    )

    # Create layer
    layer = ImprovedE3MessagePassingLayer(
        irreps_in="10x0e",
        irreps_out="32x0e + 16x1o + 8x2e",
        irreps_sh="0e + 1o + 2e",
        r_max=6.0,
        num_radial_basis=8,
        avg_num_neighbors=10,
        use_gate=True,
        use_resnet=False
    )

    # Forward pass
    output = layer(data.x, data.pos, data.edge_index)

    print(f"Input shape: {data.x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Layer has {sum(p.numel() for p in layer.parameters())} parameters")

    # Test multi-layer
    model = MultiLayerE3GNN(
        irreps_in="10x0e",
        irreps_hidden="32x0e + 16x1o + 8x2e",
        irreps_out="64x0e",
        num_layers=3
    )

    output = model(data.x, data.pos, data.edge_index)
    print(f"Multi-layer output shape: {output.shape}")
