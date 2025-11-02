#!/usr/bin/env python3
"""
E(3) Equivariant GNN Encoder - Version 2.0

Complete implementation with:
- Embedding-based input features (discrete + continuous)
- Multi-hop message passing (1-hop, 2-hop, 3-hop paths)
- Non-bonded interactions with LJ parameters
- Physical parameter integration (bonds, angles, dihedrals, LJ)
- E(3) equivariance throughout

This version is designed for RNA-3E-FFI v2.0 data format:
    Node features: [atom_type_idx, charge, residue_idx, atomic_num]
    Edges: edge_index + edge_attr (bond parameters)
    Multi-hop: triple_index/attr (angles), quadra_index/attr (dihedrals)
    Non-bonded: nonbonded_edge_index + attr (LJ parameters)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, scatter
import e3nn
from e3nn import o3
from e3nn.nn import Gate, NormActivation
from e3nn.o3 import Irreps, Linear, FullyConnectedTensorProduct
import sys
from pathlib import Path

# Add scripts to path for encoder access
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from amber_vocabulary import get_global_encoder

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


# ============================================================================
# Input Embedding Module
# ============================================================================

class AMBERFeatureEmbedding(nn.Module):
    """
    Embedding module for AMBER-based node features.

    Input: [atom_type_idx, charge, residue_idx, atomic_num] (4-dim)
    Output: Equivariant irreps features

    Strategy:
    1. Embed discrete features (atom_type_idx, residue_idx) using nn.Embedding
    2. Process continuous features (charge, atomic_num) with linear layers
    3. Fuse all features into scalar irreps
    4. Project to target irreps
    """

    def __init__(
        self,
        num_atom_types,
        num_residues,
        atom_embed_dim=32,
        residue_embed_dim=16,
        continuous_dim=16,
        output_irreps="32x0e + 16x1o + 8x2e"
    ):
        """
        Args:
            num_atom_types: Size of atom type vocabulary
            num_residues: Size of residue vocabulary
            atom_embed_dim: Embedding dimension for atom types
            residue_embed_dim: Embedding dimension for residues
            continuous_dim: Dimension for continuous features projection
            output_irreps: Target irreps for output
        """
        super().__init__()

        self.num_atom_types = num_atom_types
        self.num_residues = num_residues
        self.output_irreps = o3.Irreps(output_irreps)

        # Embedding layers for discrete features (0-indexed, including UNK)
        self.atom_type_embedding = nn.Embedding(
            num_embeddings=num_atom_types,  # Includes UNK token
            embedding_dim=atom_embed_dim
        )

        self.residue_embedding = nn.Embedding(
            num_embeddings=num_residues,  # Includes UNK token
            embedding_dim=residue_embed_dim
        )

        # Linear projection for continuous features
        self.continuous_projection = nn.Sequential(
            nn.Linear(2, continuous_dim),  # [charge, atomic_num]
            nn.SiLU(),
            nn.Linear(continuous_dim, continuous_dim)
        )

        # Feature fusion
        fusion_input_dim = atom_embed_dim + residue_embed_dim + continuous_dim

        # Extract scalar dimension from output irreps
        scalar_irreps = o3.Irreps([(mul, ir) for mul, ir in self.output_irreps if ir.l == 0])
        self.scalar_dim = scalar_irreps.dim

        # Fuse to scalar features first
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.scalar_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(self.scalar_dim * 2),
            nn.Linear(self.scalar_dim * 2, self.scalar_dim)
        )

        # Project scalars to full irreps
        scalar_irreps_str = f"{self.scalar_dim}x0e"
        self.irreps_projection = o3.Linear(
            irreps_in=o3.Irreps(scalar_irreps_str),
            irreps_out=self.output_irreps,
            internal_weights=True,
            shared_weights=True
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Node features [num_atoms, 4]
               x[:, 0] = atom_type_idx (int, 0-indexed, 0-68 normal, 69 UNK)
               x[:, 1] = charge (float)
               x[:, 2] = residue_idx (int, 0-indexed, 0-41 normal, 42 UNK)
               x[:, 3] = atomic_num (int)

        Returns:
            Embedded features [num_atoms, output_irreps_dim]
        """
        # Extract features
        atom_type_idx = x[:, 0].long()  # [num_atoms]
        charge = x[:, 1:2]              # [num_atoms, 1]
        residue_idx = x[:, 2].long()    # [num_atoms]
        atomic_num = x[:, 3:4]          # [num_atoms, 1]

        # Embed discrete features
        atom_embed = self.atom_type_embedding(atom_type_idx)  # [num_atoms, atom_embed_dim]
        residue_embed = self.residue_embedding(residue_idx)   # [num_atoms, residue_embed_dim]

        # Process continuous features
        continuous_features = torch.cat([charge, atomic_num], dim=-1)  # [num_atoms, 2]
        continuous_embed = self.continuous_projection(continuous_features)  # [num_atoms, continuous_dim]

        # Fuse all features
        fused = torch.cat([atom_embed, residue_embed, continuous_embed], dim=-1)
        scalar_features = self.feature_fusion(fused)  # [num_atoms, scalar_dim]

        # Project to irreps
        output = self.irreps_projection(scalar_features)  # [num_atoms, output_irreps_dim]

        return output


# ============================================================================
# Multi-hop Message Passing Layers
# ============================================================================

class E3GNNMessagePassingLayer(MessagePassing):
    """
    E(3) equivariant message passing layer with physical parameter integration.

    Enhanced version that accepts edge attributes (bond/LJ parameters).
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
            irreps_in: Input irreps
            irreps_out: Output irreps
            irreps_sh: Spherical harmonics irreps
            num_radial_basis: Number of radial basis functions
            radial_hidden_dim: Hidden dimension for radial MLP
            edge_attr_dim: Dimension of edge attributes (bond/LJ parameters)
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
        self.edge_attr_dim = edge_attr_dim

        # Determine if we can use residual connection
        self.use_resnet = use_resnet and (self.irreps_in == self.irreps_out)

        # Gate activation requires improved layers
        use_gate = use_gate and _has_improved_layers

        # Build gate activation
        if use_gate:
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
                irreps_before_gate = self.irreps_out
                self.nonlinearity = lambda x: x
        else:
            irreps_before_gate = self.irreps_out
            self.nonlinearity = lambda x: x

        # Pre-linear layer
        self.linear_in = Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_in,
            internal_weights=True,
            shared_weights=True
        )

        # Tensor product
        self.tp = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            irreps_before_gate,
            shared_weights=False,
            internal_weights=False
        )

        # Radial basis functions
        if _has_improved_layers:
            self.bessel_basis = BesselBasis(
                r_max=r_max,
                num_basis=num_radial_basis,
                trainable=False
            )
            self.cutoff = PolynomialCutoff(r_max=r_max, p=6)
        else:
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

        # Radial MLP (includes edge attributes)
        radial_input_dim = num_radial_basis + edge_attr_dim
        self.radial_mlp = nn.Sequential(
            nn.Linear(radial_input_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, radial_hidden_dim),
            nn.SiLU(),
            nn.Linear(radial_hidden_dim, self.tp.weight_numel)
        )

        # Self-connection (output to same space as messages before nonlinearity)
        self.self_connection = None
        if self.use_sc:
            self.self_connection = Linear(
                self.irreps_in,
                irreps_before_gate,  # Match message output dimension
                internal_weights=True,
                shared_weights=True
            )

        # Layer normalization
        self.layer_norm = None
        if self.use_layer_norm and _has_improved_layers:
            self.layer_norm = EquivariantLayerNorm(self.irreps_out)

    def forward(self, x, pos, edge_index, edge_attr=None):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, irreps_in_dim]
            pos: Node positions [num_nodes, 3]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_attr_dim] (optional)

        Returns:
            Updated node features [num_nodes, irreps_out_dim]
        """
        # Save for residual (only if irreps match after nonlinearity)
        if self.use_resnet:
            x_residual = x.clone()

        # Self-connection (applied to original input)
        if self.self_connection is not None:
            x_self = self.self_connection(x)

        # Pre-linear
        x = self.linear_in(x)

        # Message passing
        x_message = self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)

        # Average neighbors normalization
        if self.avg_num_neighbors is not None:
            x_message = x_message / (self.avg_num_neighbors ** 0.5)

        # Combine self-connection and messages
        if self.self_connection is not None:
            x = x_self + x_message
        else:
            x = x_message

        # Nonlinearity (irreps_before_gate -> irreps_out)
        x = self.nonlinearity(x)

        # Residual (add after nonlinearity)
        if self.use_resnet:
            x = x_residual + x

        # Layer norm
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x

    def message(self, x_j, pos_i, pos_j, edge_attr=None):
        """
        Construct messages.

        Args:
            x_j: Source node features [num_edges, irreps_in_dim]
            pos_i: Target positions [num_edges, 3]
            pos_j: Source positions [num_edges, 3]
            edge_attr: Edge attributes [num_edges, edge_attr_dim]

        Returns:
            Messages [num_edges, irreps_out_dim]
        """
        # Relative positions
        rel_pos = pos_i - pos_j
        distance = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
        distance = torch.clamp(distance, min=1e-6)
        rel_pos_normalized = rel_pos / distance

        # Spherical harmonics
        sh = o3.spherical_harmonics(
            self.irreps_sh,
            rel_pos_normalized,
            normalize=True,
            normalization='component'
        )

        # Radial basis
        if self.bessel_basis is not None:
            rbf = self.bessel_basis(distance)
            cutoff = self.cutoff(distance)
            rbf = rbf * cutoff
        else:
            rbf = torch.exp(-((distance - self.rbf_centers) ** 2) / (2 * self.rbf_widths ** 2))
            rbf = rbf / (torch.sqrt(2 * torch.tensor(3.14159)) * self.rbf_widths)

        # Concatenate RBF with edge attributes
        if edge_attr is not None:
            radial_input = torch.cat([rbf, edge_attr], dim=-1)
        else:
            radial_input = rbf

        # Generate tensor product weights
        tp_weights = self.radial_mlp(radial_input)

        # Apply tensor product
        messages = self.tp(x_j, sh, tp_weights)

        return messages


class AngleMessagePassing(nn.Module):
    """
    Message passing for 2-hop angle paths.

    Processes paths i -> j -> k with angle parameters.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        angle_attr_dim=2,  # [theta_eq, k]
        hidden_dim=64
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        # Extract scalar features
        scalar_irreps = o3.Irreps([(mul, ir) for mul, ir in self.irreps_in if ir.l == 0])
        self.scalar_dim = scalar_irreps.dim

        # MLP for angle feature processing
        self.angle_mlp = nn.Sequential(
            nn.Linear(self.scalar_dim * 2 + angle_attr_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.scalar_dim)
        )

        # Project back to irreps
        self.output_projection = o3.Linear(
            irreps_in=o3.Irreps(f"{self.scalar_dim}x0e"),
            irreps_out=self.irreps_out,
            internal_weights=True,
            shared_weights=True
        )

    def forward(self, x, triple_index, triple_attr):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, irreps_in_dim]
            triple_index: Angle paths [3, num_angles] (i, j, k)
            triple_attr: Angle parameters [num_angles, 2] (theta_eq, k)

        Returns:
            Angle contributions [num_nodes, irreps_out_dim]
        """
        if triple_index.shape[1] == 0:
            return torch.zeros(x.shape[0], self.irreps_out.dim, device=x.device)

        # Extract scalar features
        x_scalar = x[:, :self.scalar_dim]

        # Get node features for paths
        i, j, k = triple_index[0], triple_index[1], triple_index[2]
        x_i = x_scalar[i]  # [num_angles, scalar_dim]
        x_k = x_scalar[k]  # [num_angles, scalar_dim]

        # Concatenate features
        angle_input = torch.cat([x_i, x_k, triple_attr], dim=-1)

        # Process
        angle_messages = self.angle_mlp(angle_input)  # [num_angles, scalar_dim]

        # Aggregate to central atoms (j)
        angle_aggr = scatter(angle_messages, j, dim=0, dim_size=x.shape[0], reduce='mean')

        # Project to irreps
        output = self.output_projection(angle_aggr)

        return output


class DihedralMessagePassing(nn.Module):
    """
    Message passing for 3-hop dihedral paths.

    Processes paths i -> j -> k -> l with dihedral parameters.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        dihedral_attr_dim=3,  # [phi_k, per, phase]
        hidden_dim=64
    ):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        # Extract scalar features
        scalar_irreps = o3.Irreps([(mul, ir) for mul, ir in self.irreps_in if ir.l == 0])
        self.scalar_dim = scalar_irreps.dim

        # MLP for dihedral feature processing
        self.dihedral_mlp = nn.Sequential(
            nn.Linear(self.scalar_dim * 2 + dihedral_attr_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.scalar_dim)
        )

        # Project back to irreps
        self.output_projection = o3.Linear(
            irreps_in=o3.Irreps(f"{self.scalar_dim}x0e"),
            irreps_out=self.irreps_out,
            internal_weights=True,
            shared_weights=True
        )

    def forward(self, x, quadra_index, quadra_attr):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, irreps_in_dim]
            quadra_index: Dihedral paths [4, num_dihedrals] (i, j, k, l)
            quadra_attr: Dihedral parameters [num_dihedrals, 3] (phi_k, per, phase)

        Returns:
            Dihedral contributions [num_nodes, irreps_out_dim]
        """
        if quadra_index.shape[1] == 0:
            return torch.zeros(x.shape[0], self.irreps_out.dim, device=x.device)

        # Extract scalar features
        x_scalar = x[:, :self.scalar_dim]

        # Get node features for paths
        i, j, k, l = quadra_index[0], quadra_index[1], quadra_index[2], quadra_index[3]
        x_i = x_scalar[i]  # [num_dihedrals, scalar_dim]
        x_l = x_scalar[l]  # [num_dihedrals, scalar_dim]

        # Concatenate features
        dihedral_input = torch.cat([x_i, x_l, quadra_attr], dim=-1)

        # Process
        dihedral_messages = self.dihedral_mlp(dihedral_input)  # [num_dihedrals, scalar_dim]

        # Aggregate to central atoms (j and k)
        # We aggregate to both j and k and average
        dihedral_aggr_j = scatter(dihedral_messages, j, dim=0, dim_size=x.shape[0], reduce='mean')
        dihedral_aggr_k = scatter(dihedral_messages, k, dim=0, dim_size=x.shape[0], reduce='mean')
        dihedral_aggr = (dihedral_aggr_j + dihedral_aggr_k) / 2

        # Project to irreps
        output = self.output_projection(dihedral_aggr)

        return output


# ============================================================================
# Main Model
# ============================================================================

class RNAPocketEncoderV2(nn.Module):
    """
    E(3) Equivariant GNN for RNA binding pockets - Version 2.0

    Complete implementation with:
    - Embedding-based input features
    - Multi-hop message passing (1-hop bonds, 2-hop angles, 3-hop dihedrals)
    - Non-bonded interactions with LJ parameters
    - Physical parameter integration
    """

    def __init__(
        self,
        num_atom_types=71,  # AMBER vocabulary size + 1 for <UNK>
        num_residues=43,    # RNA residue vocabulary size + 1 for <UNK>
        atom_embed_dim=32,
        residue_embed_dim=16,
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
        use_multi_hop=True,
        use_nonbonded=True,
        pooling_type='attention',
        dropout=0.0
    ):
        """
        Args:
            num_atom_types: Size of atom type vocabulary
            num_residues: Size of residue vocabulary
            atom_embed_dim: Embedding dimension for atom types
            residue_embed_dim: Embedding dimension for residues
            hidden_irreps: Hidden layer irreps
            output_dim: Final pocket embedding dimension
            num_layers: Number of message passing layers
            num_radial_basis: Number of radial basis functions
            radial_hidden_dim: Hidden dimension for radial MLPs
            pooling_hidden_dim: Hidden dimension for pooling MLP
            r_max: Maximum cutoff distance
            avg_num_neighbors: Average number of neighbors
            use_gate: Whether to use gate activation
            use_layer_norm: Whether to use layer normalization
            use_multi_hop: Whether to use multi-hop paths
            use_nonbonded: Whether to use non-bonded edges
            pooling_type: Type of pooling ('attention', 'mean', 'sum', 'max')
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.use_multi_hop = use_multi_hop
        self.use_nonbonded = use_nonbonded

        # Learnable combining weights for multi-hop and non-bonded contributions
        # Use log-space parameters to ensure weights stay positive and bounded
        if use_multi_hop:
            self.angle_weight_raw = nn.Parameter(torch.log(torch.tensor(0.333)))
            self.dihedral_weight_raw = nn.Parameter(torch.log(torch.tensor(0.333)))
        if use_nonbonded:
            self.nonbonded_weight_raw = nn.Parameter(torch.log(torch.tensor(0.333)))

        # Input embedding
        self.input_embedding = AMBERFeatureEmbedding(
            num_atom_types=num_atom_types,
            num_residues=num_residues,
            atom_embed_dim=atom_embed_dim,
            residue_embed_dim=residue_embed_dim,
            continuous_dim=16,
            output_irreps=hidden_irreps
        )

        # 1-hop message passing layers (bonded)
        self.bonded_mp_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = E3GNNMessagePassingLayer(
                irreps_in=self.hidden_irreps,
                irreps_out=self.hidden_irreps,
                irreps_sh="0e + 1o + 2e",
                num_radial_basis=num_radial_basis,
                radial_hidden_dim=radial_hidden_dim,
                edge_attr_dim=2,  # [req, k] for bonds
                r_max=r_max,
                avg_num_neighbors=avg_num_neighbors,
                use_gate=use_gate,
                use_sc=True,
                use_resnet=True,
                use_layer_norm=use_layer_norm
            )
            self.bonded_mp_layers.append(layer)

        # 2-hop angle message passing (if enabled)
        if use_multi_hop:
            self.angle_mp_layers = nn.ModuleList()
            for i in range(num_layers):
                layer = AngleMessagePassing(
                    irreps_in=self.hidden_irreps,
                    irreps_out=self.hidden_irreps,
                    angle_attr_dim=2,  # [theta_eq, k]
                    hidden_dim=64
                )
                self.angle_mp_layers.append(layer)

        # 3-hop dihedral message passing (if enabled)
        if use_multi_hop:
            self.dihedral_mp_layers = nn.ModuleList()
            for i in range(num_layers):
                layer = DihedralMessagePassing(
                    irreps_in=self.hidden_irreps,
                    irreps_out=self.hidden_irreps,
                    dihedral_attr_dim=3,  # [phi_k, per, phase]
                    hidden_dim=64
                )
                self.dihedral_mp_layers.append(layer)

        # Non-bonded message passing (if enabled)
        if use_nonbonded:
            self.nonbonded_mp_layers = nn.ModuleList()
            for i in range(num_layers):
                layer = E3GNNMessagePassingLayer(
                    irreps_in=self.hidden_irreps,
                    irreps_out=self.hidden_irreps,
                    irreps_sh="0e + 1o + 2e",
                    num_radial_basis=num_radial_basis,
                    radial_hidden_dim=radial_hidden_dim,
                    edge_attr_dim=3,  # [LJ_A, LJ_B, distance]
                    r_max=r_max,
                    avg_num_neighbors=avg_num_neighbors,
                    use_gate=use_gate,
                    use_sc=False,  # No self-connection for non-bonded
                    use_resnet=False,
                    use_layer_norm=use_layer_norm
                )
                self.nonbonded_mp_layers.append(layer)

        # Extract scalar dimension
        scalar_irreps = o3.Irreps([(mul, ir) for mul, ir in self.hidden_irreps if ir.l == 0])
        self.scalar_dim = scalar_irreps.dim

        # Calculate invariant representation dimension
        # invariant_dim = scalar_dim + number of l=1 irreps + number of l=2 irreps
        # For "32x0e + 16x1o + 8x2e": invariant_dim = 32 + 16 + 8 = 56
        # We take L2 norm of each vector (l=1) and tensor (l=2) to get scalar invariants
        self.num_l1_irreps = sum(mul for mul, ir in self.hidden_irreps if ir.l == 1)
        self.num_l2_irreps = sum(mul for mul, ir in self.hidden_irreps if ir.l == 2)
        self.invariant_dim = self.scalar_dim + self.num_l1_irreps + self.num_l2_irreps

        # Build index ranges for each irrep type
        self._build_irreps_slices()

        # Pooling (uses invariant representation)
        if pooling_type == 'attention':
            self.pooling_mlp = nn.Sequential(
                nn.Linear(self.invariant_dim, pooling_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(pooling_hidden_dim, pooling_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(pooling_hidden_dim, 1)
            )
        else:
            self.pooling_mlp = None

        # Output projection (uses invariant representation)
        # Note: Always use LayerNorm at output to prevent numerical instability
        # Even if use_layer_norm=False for message passing layers
        self.output_projection = nn.Sequential(
            nn.Linear(self.invariant_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def get_angle_weight(self):
        """Get angle weight (ensures it stays positive and bounded)."""
        # Clamp to prevent extreme values: exp(-5) ≈ 0.0067, exp(5) ≈ 148
        return torch.exp(torch.clamp(self.angle_weight_raw, min=-5, max=5))

    def get_dihedral_weight(self):
        """Get dihedral weight (ensures it stays positive and bounded)."""
        return torch.exp(torch.clamp(self.dihedral_weight_raw, min=-5, max=5))

    def get_nonbonded_weight(self):
        """Get nonbonded weight (ensures it stays positive and bounded)."""
        return torch.exp(torch.clamp(self.nonbonded_weight_raw, min=-5, max=5))

    def _build_irreps_slices(self):
        """
        Build index slices for extracting different irrep types.
        This is used to efficiently extract scalars, vectors, and tensors.
        """
        self.irreps_slices = {'l0': [], 'l1': [], 'l2': []}

        idx = 0
        for mul, ir in self.hidden_irreps:
            dim = ir.dim
            for _ in range(mul):
                if ir.l == 0:
                    self.irreps_slices['l0'].append((idx, idx + dim))
                elif ir.l == 1:
                    self.irreps_slices['l1'].append((idx, idx + dim))
                elif ir.l == 2:
                    self.irreps_slices['l2'].append((idx, idx + dim))
                idx += dim

    def extract_invariant_features(self, h):
        """
        Extract E(3) invariant features from equivariant representations.

        Since molecular fingerprints and Uni-Mol embeddings are E3 invariant,
        we need to extract invariant features from the equivariant representations.

        Strategy:
        1. Extract scalar features (l=0, even) directly - these are already invariant
        2. Compute L2 norm of each vector (l=1, odd) - gives rotation invariant magnitude
        3. Compute L2 norm of each 2nd-order tensor (l=2, even) - gives rotation invariant magnitude
        4. Concatenate all invariant features

        Args:
            h: Equivariant features [num_atoms, hidden_irreps_dim]

        Returns:
            t: Invariant features [num_atoms, invariant_dim]
               invariant_dim = num_scalars + num_l1_irreps + num_l2_irreps

        Example:
            For "32x0e + 16x1o + 8x2e":
            - 32 scalars (l=0): used directly → 32 features
            - 16 vectors (l=1): each 3D vector → 1 scalar (L2 norm) → 16 features
            - 8 tensors (l=2): each 5D tensor → 1 scalar (L2 norm) → 8 features
            - Total invariant_dim = 32 + 16 + 8 = 56
        """
        device = h.device
        num_atoms = h.shape[0]

        # Collect invariant features
        invariant_features = []

        # 1. Extract scalars (l=0) - already invariant
        for start, end in self.irreps_slices['l0']:
            invariant_features.append(h[:, start:end])

        # 2. L2 norm of vectors (l=1)
        for start, end in self.irreps_slices['l1']:
            vec = h[:, start:end]  # [num_atoms, 3]
            norm = torch.linalg.norm(vec, dim=-1, keepdim=True)  # [num_atoms, 1]
            invariant_features.append(norm)

        # 3. L2 norm of 2nd-order tensors (l=2)
        for start, end in self.irreps_slices['l2']:
            tensor = h[:, start:end]  # [num_atoms, 5]
            norm = torch.linalg.norm(tensor, dim=-1, keepdim=True)  # [num_atoms, 1]
            invariant_features.append(norm)

        # Concatenate all invariant features
        t = torch.cat(invariant_features, dim=-1)  # [num_atoms, invariant_dim]

        return t

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_atoms, 4]
                - pos: Node positions [num_atoms, 3]
                - edge_index: Bonded edges [2, num_bonds]
                - edge_attr: Bond parameters [num_bonds, 2]
                - triple_index: Angle paths [3, num_angles] (optional)
                - triple_attr: Angle parameters [num_angles, 2] (optional)
                - quadra_index: Dihedral paths [4, num_dihedrals] (optional)
                - quadra_attr: Dihedral parameters [num_dihedrals, 3] (optional)
                - nonbonded_edge_index: Non-bonded edges [2, num_nb] (optional)
                - nonbonded_edge_attr: LJ parameters [num_nb, 3] (optional)
                - batch: Batch indices [num_atoms] (optional)

        Returns:
            Pocket embeddings [batch_size, output_dim]
        """
        x, pos, edge_index = data.x, data.pos, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        # Get batch
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Initial embedding
        h = self.input_embedding(x)

        # Message passing layers
        for i in range(self.num_layers):
            h_new = h

            # 1-hop bonded message passing
            h_bonded = self.bonded_mp_layers[i](h, pos, edge_index, edge_attr)
            h_new = h_bonded

            # 2-hop angle message passing (learnable weight)
            if self.use_multi_hop and hasattr(data, 'triple_index'):
                h_angle = self.angle_mp_layers[i](h, data.triple_index, data.triple_attr)
                h_new = h_new + self.get_angle_weight() * h_angle

            # 3-hop dihedral message passing (learnable weight)
            if self.use_multi_hop and hasattr(data, 'quadra_index'):
                h_dihedral = self.dihedral_mp_layers[i](h, data.quadra_index, data.quadra_attr)
                h_new = h_new + self.get_dihedral_weight() * h_dihedral

            # Non-bonded message passing (learnable weight)
            if self.use_nonbonded and hasattr(data, 'nonbonded_edge_index'):
                h_nonbonded = self.nonbonded_mp_layers[i](
                    h, pos, data.nonbonded_edge_index, data.nonbonded_edge_attr
                )
                h_new = h_new + self.get_nonbonded_weight() * h_nonbonded

            h = h_new

            # Dropout on scalars
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

        # Extract invariant features (scalars + L2 norms of higher-order tensors)
        # This is crucial for compatibility with E3-invariant ligand representations
        # (molecular fingerprints, Uni-Mol embeddings)
        t = self.extract_invariant_features(h)

        # Pooling
        if self.pooling_type == 'attention' and self.pooling_mlp is not None:
            attention_logits = self.pooling_mlp(t)
            attention_weights = softmax(attention_logits, index=batch, dim=0)
            weighted_features = t * attention_weights
            graph_embedding = scatter(
                weighted_features,
                index=batch,
                dim=0,
                reduce='sum'
            )
        elif self.pooling_type == 'mean':
            graph_embedding = scatter(t, index=batch, dim=0, reduce='mean')
        elif self.pooling_type == 'sum':
            graph_embedding = scatter(t, index=batch, dim=0, reduce='sum')
        elif self.pooling_type == 'max':
            graph_embedding = scatter(t, index=batch, dim=0, reduce='max')
        else:
            # Default to mean
            graph_embedding = scatter(t, index=batch, dim=0, reduce='mean')

        # Project to output
        output = self.output_projection(graph_embedding)

        return output

    def get_node_embeddings(self, data):
        """
        Get per-node invariant embeddings (before pooling).

        Returns invariant features t_i for each atom, which includes:
        - Scalar features (l=0)
        - L2 norms of vectors (l=1)
        - L2 norms of tensors (l=2)

        This ensures compatibility with E3-invariant ligand embeddings.
        """
        x, pos, edge_index = data.x, data.pos, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        # Initial embedding
        h = self.input_embedding(x)

        # Message passing
        for i in range(self.num_layers):
            h_bonded = self.bonded_mp_layers[i](h, pos, edge_index, edge_attr)
            h_new = h_bonded

            if self.use_multi_hop and hasattr(data, 'triple_index'):
                h_angle = self.angle_mp_layers[i](h, data.triple_index, data.triple_attr)
                h_new = h_new + self.get_angle_weight() * h_angle

            if self.use_multi_hop and hasattr(data, 'quadra_index'):
                h_dihedral = self.dihedral_mp_layers[i](h, data.quadra_index, data.quadra_attr)
                h_new = h_new + self.get_dihedral_weight() * h_dihedral

            if self.use_nonbonded and hasattr(data, 'nonbonded_edge_index'):
                h_nonbonded = self.nonbonded_mp_layers[i](
                    h, pos, data.nonbonded_edge_index, data.nonbonded_edge_attr
                )
                h_new = h_new + self.get_nonbonded_weight() * h_nonbonded

            h = h_new

        # Extract invariant features
        t = self.extract_invariant_features(h)

        return t


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    print("=" * 80)
    print("Testing E(3) GNN Encoder v2.0")
    print("=" * 80)

    # Get encoder for vocabulary sizes
    encoder = get_global_encoder()

    # Create dummy data matching v2.0 format
    num_nodes = 100
    num_edges = 300
    num_angles = 150
    num_dihedrals = 80
    num_nonbonded = 200

    # Create realistic test data (0-indexed)
    x = torch.zeros(num_nodes, 4)
    x[:, 0] = torch.randint(0, encoder.num_atom_types, (num_nodes,)).float()  # atom_type_idx (0-indexed)
    x[:, 1] = torch.randn(num_nodes) * 0.5  # charge
    x[:, 2] = torch.randint(0, encoder.num_residues, (num_nodes,)).float()  # residue_idx (0-indexed)
    x[:, 3] = torch.randint(1, 20, (num_nodes,)).float()  # atomic_num (1-19 common elements)

    data = Data(
        x=x,
        pos=torch.randn(num_nodes, 3),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_attr=torch.randn(num_edges, 2).abs() + 0.1,  # [req, k] positive values
        triple_index=torch.randint(0, num_nodes, (3, num_angles)),
        triple_attr=torch.randn(num_angles, 2).abs() + 0.1,  # [theta_eq, k] positive
        quadra_index=torch.randint(0, num_nodes, (4, num_dihedrals)),
        quadra_attr=torch.randn(num_dihedrals, 3),  # [phi_k, per, phase]
        nonbonded_edge_index=torch.randint(0, num_nodes, (2, num_nonbonded)),
        nonbonded_edge_attr=torch.cat([
            torch.randn(num_nonbonded, 2).abs(),  # LJ_A, LJ_B positive
            torch.rand(num_nonbonded, 1) * 6.0  # distance 0-6 Angstroms
        ], dim=-1)
    )

    # Create model
    model = RNAPocketEncoderV2(
        num_atom_types=encoder.num_atom_types,
        num_residues=encoder.num_residues,
        hidden_irreps="32x0e + 16x1o + 8x2e",
        output_dim=512,
        num_layers=3,
        use_gate=True,
        use_layer_norm=True,
        use_multi_hop=True,
        use_nonbonded=True,
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
