#!/usr/bin/env python3
"""
Radial Basis Functions and Cutoff Functions

Implements Bessel basis functions and polynomial cutoff for distance encoding.
Based on DimeNet (https://arxiv.org/abs/2003.03123)
"""
import math
import torch
import torch.nn as nn


class PolynomialCutoff(nn.Module):
    """
    Polynomial cutoff function for smooth distance truncation.

    The cutoff ensures smooth decay to zero at r_max, which is crucial
    for maintaining smooth gradients in force predictions.

    cutoff(r) = 1 - ((p+1)(p+2)/2)(r/r_max)^p + p(p+2)(r/r_max)^(p+1) - (p(p+1)/2)(r/r_max)^(p+2)

    Args:
        r_max: Maximum cutoff distance
        p: Polynomial degree (default: 6)
    """

    def __init__(self, r_max: float, p: float = 6.0):
        super().__init__()
        assert p >= 2.0, "Polynomial degree must be at least 2"
        self.p = float(p)
        self.r_max = float(r_max)
        self._factor = 1.0 / self.r_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cutoff function.

        Args:
            x: Distance tensor [*, 1] or [*]

        Returns:
            Cutoff values in [0, 1]
        """
        # Normalize by r_max
        x_normalized = x * self._factor

        # Compute polynomial cutoff
        p = self.p
        out = 1.0
        out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x_normalized, p))
        out = out + (p * (p + 2.0) * torch.pow(x_normalized, p + 1.0))
        out = out - ((p * (p + 1.0) / 2.0) * torch.pow(x_normalized, p + 2.0))

        # Set to 0 for distances >= r_max
        out = out * (x_normalized < 1.0).float()

        return out


class BesselBasis(nn.Module):
    """
    Bessel radial basis functions.

    Uses sine-based Bessel functions for distance encoding:
    φ_n(r) = sqrt(2/(r_max - r_min)) * sin(nπr/(r_max - r_min)) / r

    The 1/r factor makes it suitable for Coulomb-like interactions.

    Args:
        r_max: Maximum distance
        r_min: Minimum distance (default: 0)
        num_basis: Number of basis functions
        trainable: Whether to make the frequencies trainable

    Reference:
        DimeNet: https://arxiv.org/abs/2003.03123
    """

    def __init__(
        self,
        r_max: float,
        r_min: float = 0.0,
        num_basis: int = 8,
        trainable: bool = False
    ):
        super().__init__()

        self.r_max = float(r_max)
        self.r_min = float(r_min)
        self.num_basis = num_basis
        self.trainable = trainable

        # Normalization factor: sqrt(2 / (r_max - r_min))
        self.prefactor = math.sqrt(2.0 / (self.r_max - self.r_min))

        # Frequencies: n * π for n = 1, 2, ..., num_basis
        frequencies = torch.linspace(1.0, num_basis, num_basis) * math.pi

        if trainable:
            self.frequencies = nn.Parameter(frequencies)
        else:
            self.register_buffer('frequencies', frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Bessel basis functions.

        Args:
            x: Distance tensor [num_edges, 1] or [num_edges]

        Returns:
            Basis function values [num_edges, num_basis]
        """
        # Ensure x is 2D
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Avoid division by zero
        x = torch.clamp(x, min=1e-8)

        # Compute: sin(freq * r / (r_max - r_min)) / r
        # Shape: [num_edges, 1] * [num_basis] -> [num_edges, num_basis]
        arg = self.frequencies * x / (self.r_max - self.r_min)
        numerator = torch.sin(arg)

        # Apply 1/r factor and normalization
        result = self.prefactor * numerator / x

        return result


class GaussianBasis(nn.Module):
    """
    Gaussian radial basis functions (for comparison).

    φ_n(r) = exp(-γ_n * (r - μ_n)^2)

    Args:
        r_max: Maximum distance
        num_basis: Number of basis functions
        learnable: Whether centers and widths are learnable
    """

    def __init__(
        self,
        r_max: float,
        num_basis: int = 8,
        learnable: bool = False
    ):
        super().__init__()

        self.r_max = r_max
        self.num_basis = num_basis

        # Initialize centers uniformly from 0 to r_max
        centers = torch.linspace(0, r_max, num_basis)

        # Initialize widths
        widths = torch.ones(num_basis) * 0.5

        if learnable:
            self.centers = nn.Parameter(centers)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('widths', widths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian basis functions.

        Args:
            x: Distance tensor [num_edges, 1] or [num_edges]

        Returns:
            Basis function values [num_edges, num_basis]
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Gaussian RBF: exp(-((x - μ) / σ)^2 / 2)
        diff = x - self.centers
        gaussian = torch.exp(-0.5 * (diff / self.widths) ** 2)

        # Normalize
        gaussian = gaussian / (math.sqrt(2 * math.pi) * self.widths)

        return gaussian


# Example usage and comparison
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test Bessel basis
    r_max = 6.0
    num_basis = 8

    bessel = BesselBasis(r_max=r_max, num_basis=num_basis)
    gaussian = GaussianBasis(r_max=r_max, num_basis=num_basis)
    cutoff = PolynomialCutoff(r_max=r_max, p=6)

    # Generate distances
    distances = torch.linspace(0.1, r_max + 1, 200).unsqueeze(-1)

    # Compute basis functions
    bessel_values = bessel(distances)
    gaussian_values = gaussian(distances)
    cutoff_values = cutoff(distances)

    print(f"Bessel basis shape: {bessel_values.shape}")
    print(f"Gaussian basis shape: {gaussian_values.shape}")
    print(f"Cutoff shape: {cutoff_values.shape}")

    # Test with batch
    batch_distances = torch.randn(100, 1).abs() * r_max
    batch_bessel = bessel(batch_distances)
    print(f"Batch Bessel shape: {batch_bessel.shape}")
