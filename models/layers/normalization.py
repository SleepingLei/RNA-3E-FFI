#!/usr/bin/env python3
"""
Normalization layers for E(3) equivariant networks.

Implements layer normalization that respects the structure of irreducible representations.
"""
import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import Irreps


class EquivariantLayerNorm(nn.Module):
    """
    Layer normalization for irreps that preserves equivariance.

    Normalizes each irrep component separately:
    - For scalars (l=0): standard normalization
    - For vectors/tensors (l>0): normalize by norm

    Args:
        irreps: Irreducible representations
        eps: Small constant for numerical stability
        affine: Whether to learn affine parameters
        normalization: Type of normalization ('component' or 'norm')
    """

    def __init__(
        self,
        irreps,
        eps: float = 1e-5,
        affine: bool = True,
        normalization: str = 'component'
    ):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine
        self.normalization = normalization

        # Separate scalar and non-scalar irreps
        self.scalar_indices = []
        self.nonscalar_indices = []

        idx = 0
        for mul, ir in self.irreps:
            dim = ir.dim
            for _ in range(mul):
                if ir.l == 0:
                    self.scalar_indices.append((idx, idx + dim))
                else:
                    self.nonscalar_indices.append((idx, idx + dim, ir.l))
                idx += dim

        # Affine parameters for scalars
        if affine and len(self.scalar_indices) > 0:
            num_scalars = sum(end - start for start, end in self.scalar_indices)
            self.weight = nn.Parameter(torch.ones(num_scalars))
            self.bias = nn.Parameter(torch.zeros(num_scalars))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply equivariant layer normalization.

        Args:
            x: Input tensor [batch, irreps_dim]

        Returns:
            Normalized tensor [batch, irreps_dim]
        """
        if x.size(-1) != self.irreps.dim:
            raise ValueError(
                f"Input dimension {x.size(-1)} does not match irreps dimension {self.irreps.dim}"
            )

        output = x.clone()

        # Normalize scalars (l=0)
        if len(self.scalar_indices) > 0:
            scalar_parts = []
            for start, end in self.scalar_indices:
                scalar_parts.append(x[..., start:end])

            if len(scalar_parts) > 0:
                scalars = torch.cat(scalar_parts, dim=-1)

                # Standard layer normalization
                mean = scalars.mean(dim=-1, keepdim=True)
                var = scalars.var(dim=-1, keepdim=True, unbiased=False)
                scalars_normalized = (scalars - mean) / torch.sqrt(var + self.eps)

                # Apply affine transformation
                if self.affine and self.weight is not None:
                    scalars_normalized = scalars_normalized * self.weight + self.bias

                # Put normalized scalars back
                offset = 0
                for start, end in self.scalar_indices:
                    length = end - start
                    output[..., start:end] = scalars_normalized[..., offset:offset + length]
                    offset += length

        # Normalize non-scalars (l>0) by their norm
        for start, end, l in self.nonscalar_indices:
            vec = x[..., start:end]

            if self.normalization == 'norm':
                # Reshape to [batch, 2l+1]
                dim = 2 * l + 1
                vec_reshaped = vec.view(*vec.shape[:-1], -1, dim)

                # Compute norm
                norm = torch.linalg.norm(vec_reshaped, dim=-1, keepdim=True)
                norm = torch.clamp(norm, min=self.eps)

                # Normalize
                vec_normalized = vec_reshaped / norm
                output[..., start:end] = vec_normalized.view(*vec.shape)
            else:
                # Component-wise normalization
                mean = vec.mean(dim=-1, keepdim=True)
                var = vec.var(dim=-1, keepdim=True, unbiased=False)
                vec_normalized = (vec - mean) / torch.sqrt(var + self.eps)
                output[..., start:end] = vec_normalized

        return output


class EquivariantRMSNorm(nn.Module):
    """
    RMS Normalization for irreps (faster alternative to LayerNorm).

    Only normalizes by RMS without mean centering.

    Args:
        irreps: Irreducible representations
        eps: Small constant for numerical stability
        affine: Whether to learn scale parameter
    """

    def __init__(
        self,
        irreps,
        eps: float = 1e-5,
        affine: bool = True
    ):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine

        if affine:
            self.scale = nn.Parameter(torch.ones(self.irreps.dim))
        else:
            self.register_parameter('scale', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor [batch, irreps_dim]

        Returns:
            Normalized tensor [batch, irreps_dim]
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize
        output = x / rms

        # Apply learned scale
        if self.affine and self.scale is not None:
            output = output * self.scale

        return output


class EquivariantInstanceNorm(nn.Module):
    """
    Instance normalization for irreps.

    Normalizes each sample independently, useful for varying batch sizes.

    Args:
        irreps: Irreducible representations
        eps: Small constant for numerical stability
        affine: Whether to learn affine parameters
    """

    def __init__(
        self,
        irreps,
        eps: float = 1e-5,
        affine: bool = True
    ):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.affine = affine

        # Count scalar dimensions
        self.num_scalars = sum(
            mul * ir.dim for mul, ir in self.irreps if ir.l == 0
        )

        if affine and self.num_scalars > 0:
            self.weight = nn.Parameter(torch.ones(self.num_scalars))
            self.bias = nn.Parameter(torch.zeros(self.num_scalars))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply instance normalization.

        Args:
            x: Input tensor [batch, irreps_dim]

        Returns:
            Normalized tensor [batch, irreps_dim]
        """
        output = x.clone()

        # Extract scalar features
        idx = 0
        scalar_start = 0
        scalars = []

        for mul, ir in self.irreps:
            dim = mul * ir.dim
            if ir.l == 0:
                scalars.append(x[..., idx:idx + dim])
            idx += dim

        if len(scalars) > 0:
            scalars = torch.cat(scalars, dim=-1)

            # Instance norm: normalize each sample
            mean = scalars.mean(dim=-1, keepdim=True)
            var = scalars.var(dim=-1, keepdim=True, unbiased=False)
            scalars_normalized = (scalars - mean) / torch.sqrt(var + self.eps)

            # Apply affine
            if self.affine and self.weight is not None:
                scalars_normalized = scalars_normalized * self.weight + self.bias

            # Put back
            idx = 0
            scalar_offset = 0
            for mul, ir in self.irreps:
                dim = mul * ir.dim
                if ir.l == 0:
                    output[..., idx:idx + dim] = scalars_normalized[
                        ..., scalar_offset:scalar_offset + dim
                    ]
                    scalar_offset += dim
                idx += dim

        return output


# Example usage
if __name__ == "__main__":
    # Test with mixed irreps
    irreps = o3.Irreps("32x0e + 16x1o + 8x2e")
    batch_size = 10

    # Create random data
    x = torch.randn(batch_size, irreps.dim)

    # Test LayerNorm
    ln = EquivariantLayerNorm(irreps)
    x_normalized = ln(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_normalized.shape}")
    print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"Output mean: {x_normalized.mean():.4f}, std: {x_normalized.std():.4f}")

    # Test RMSNorm
    rms = EquivariantRMSNorm(irreps)
    x_rms = rms(x)
    print(f"RMS normalized mean: {x_rms.mean():.4f}, std: {x_rms.std():.4f}")

    # Check equivariance for non-scalar parts
    # Rotation should preserve the normalization
    from e3nn.o3 import rand_matrix
    R = rand_matrix()
    D = irreps.D_from_matrix(R)

    x_rotated = x @ D.T
    x_rotated_normalized = ln(x_rotated)
    x_normalized_rotated = ln(x) @ D.T

    diff = torch.abs(x_rotated_normalized - x_normalized_rotated).max()
    print(f"Equivariance error: {diff:.6f}")
