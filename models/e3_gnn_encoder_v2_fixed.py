#!/usr/bin/env python3
"""
E(3) Equivariant GNN Encoder - Version 2.0 with Weight Constraints

This version adds constraints to learnable weights to prevent them from
becoming zero during training. Uses log-space parameterization:
    weight = exp(log_weight)

This ensures weights are always positive (weight > 0) and can never reach zero.

**When to use this version:**
- When training with incomplete multi-hop data
- As a safety measure to prevent weight collapse
- When you want stable weight learning

**Mathematical advantage:**
- Standard: weight can go to 0 if gradients are negative
- Log-space: weight = exp(log_weight) is always > 0
- Gradient flow: d(weight)/d(log_weight) = weight (natural gradient)

Inherits from RNAPocketEncoderV2 and overrides weight initialization.
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Handle imports for both module and script usage
try:
    from .e3_gnn_encoder_v2 import RNAPocketEncoderV2
except ImportError:
    # Running as script
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.e3_gnn_encoder_v2 import RNAPocketEncoderV2


class RNAPocketEncoderV2Fixed(RNAPocketEncoderV2):
    """
    RNAPocketEncoderV2 with weight constraints.

    Uses log-space parameterization to ensure weights remain positive:
        - Stores: log_weight (unconstrained)
        - Returns: exp(log_weight) (always > 0)

    Example:
        >>> model = RNAPocketEncoderV2Fixed(
        ...     num_atom_types=71,
        ...     num_residues=43,
        ...     use_multi_hop=True,
        ...     use_nonbonded=True
        ... )
        >>>
        >>> # Weights are always positive
        >>> print(model.angle_weight)  # Always > 0
        >>>
        >>> # Even if gradients push weights down, they can't reach 0
        >>> # because exp(x) > 0 for all x
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize model with constrained weights.

        All arguments are passed to parent RNAPocketEncoderV2.
        After initialization, replaces standard weights with log-space versions.
        """
        # Initialize parent model
        super().__init__(*args, **kwargs)

        # Replace standard weights with log-space versions
        if self.use_multi_hop:
            # Store initial values
            angle_init = self.angle_weight.data.clone()
            dihedral_init = self.dihedral_weight.data.clone()

            # Delete standard parameters
            del self.angle_weight
            del self.dihedral_weight

            # Create log-space parameters
            # log(0.5) ≈ -0.693, log(0.3) ≈ -1.204
            self.angle_log_weight = nn.Parameter(angle_init.log())
            self.dihedral_log_weight = nn.Parameter(dihedral_init.log())

        if self.use_nonbonded:
            # Store initial value
            nonbonded_init = self.nonbonded_weight.data.clone()

            # Delete standard parameter
            del self.nonbonded_weight

            # Create log-space parameter
            # log(0.2) ≈ -1.609
            self.nonbonded_log_weight = nn.Parameter(nonbonded_init.log())

    @property
    def angle_weight(self):
        """
        Get angle weight (always positive).

        Returns:
            torch.Tensor: exp(log_weight), guaranteed > 0

        Example:
            >>> model = RNAPocketEncoderV2Fixed(use_multi_hop=True)
            >>> w = model.angle_weight
            >>> assert w > 0  # Always true
        """
        if not self.use_multi_hop:
            return None
        return self.angle_log_weight.exp()

    @property
    def dihedral_weight(self):
        """
        Get dihedral weight (always positive).

        Returns:
            torch.Tensor: exp(log_weight), guaranteed > 0
        """
        if not self.use_multi_hop:
            return None
        return self.dihedral_log_weight.exp()

    @property
    def nonbonded_weight(self):
        """
        Get non-bonded weight (always positive).

        Returns:
            torch.Tensor: exp(log_weight), guaranteed > 0
        """
        if not self.use_nonbonded:
            return None
        return self.nonbonded_log_weight.exp()

    def get_weight_summary(self):
        """
        Get current weight values for monitoring.

        Returns:
            dict: Weight values and their log-space parameters

        Example:
            >>> summary = model.get_weight_summary()
            >>> print(summary)
            {
                'angle_weight': 0.4823,
                'angle_log_weight': -0.7291,
                'dihedral_weight': 0.3156,
                'dihedral_log_weight': -1.1543,
                'nonbonded_weight': 0.1891,
                'nonbonded_log_weight': -1.6653
            }
        """
        summary = {}

        if self.use_multi_hop:
            summary['angle_weight'] = self.angle_weight.item()
            summary['angle_log_weight'] = self.angle_log_weight.item()
            summary['dihedral_weight'] = self.dihedral_weight.item()
            summary['dihedral_log_weight'] = self.dihedral_log_weight.item()

        if self.use_nonbonded:
            summary['nonbonded_weight'] = self.nonbonded_weight.item()
            summary['nonbonded_log_weight'] = self.nonbonded_log_weight.item()

        return summary

    def __repr__(self):
        """String representation with weight information."""
        base_repr = super().__repr__()

        weight_info = "\nLearnable Weights (Constrained):"
        if self.use_multi_hop:
            weight_info += f"\n  angle_weight: {self.angle_weight.item():.4f} (log: {self.angle_log_weight.item():.4f})"
            weight_info += f"\n  dihedral_weight: {self.dihedral_weight.item():.4f} (log: {self.dihedral_log_weight.item():.4f})"
        if self.use_nonbonded:
            weight_info += f"\n  nonbonded_weight: {self.nonbonded_weight.item():.4f} (log: {self.nonbonded_log_weight.item():.4f})"

        return base_repr + weight_info


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    """
    Example demonstrating the weight constraint mechanism.
    """
    print("=" * 80)
    print("RNAPocketEncoderV2Fixed - Weight Constraint Demo")
    print("=" * 80)

    # Create model
    model = RNAPocketEncoderV2Fixed(
        num_atom_types=71,
        num_residues=43,
        hidden_irreps="16x0e + 8x1o",
        output_dim=128,
        num_layers=2,
        use_multi_hop=True,
        use_nonbonded=True
    )

    print("\n1. Initial weights:")
    print(f"   angle_weight: {model.angle_weight.item():.6f}")
    print(f"   dihedral_weight: {model.dihedral_weight.item():.6f}")
    print(f"   nonbonded_weight: {model.nonbonded_weight.item():.6f}")

    # Simulate training with negative gradients
    print("\n2. Simulating 100 steps with strong negative gradients...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for step in range(100):
        # Manually set strong negative gradients (simulating weight collapse)
        model.angle_log_weight.grad = torch.tensor(-1.0)
        model.dihedral_log_weight.grad = torch.tensor(-1.0)
        model.nonbonded_log_weight.grad = torch.tensor(-1.0)

        # Update
        optimizer.step()
        optimizer.zero_grad()

        if step % 20 == 0:
            print(f"\n   Step {step}:")
            print(f"     angle_weight: {model.angle_weight.item():.6f}")
            print(f"     dihedral_weight: {model.dihedral_weight.item():.6f}")
            print(f"     nonbonded_weight: {model.nonbonded_weight.item():.6f}")

    print("\n3. Final weights after 100 steps:")
    print(f"   angle_weight: {model.angle_weight.item():.6f} (still > 0!)")
    print(f"   dihedral_weight: {model.dihedral_weight.item():.6f} (still > 0!)")
    print(f"   nonbonded_weight: {model.nonbonded_weight.item():.6f} (still > 0!)")

    print("\n4. Log-space parameters:")
    summary = model.get_weight_summary()
    for key, value in summary.items():
        if 'log' in key:
            print(f"   {key}: {value:.6f}")

    print("\n" + "=" * 80)
    print("✅ Weights remain positive even with strong negative gradients!")
    print("=" * 80)

    # Comparison with standard version
    print("\n5. Comparison: Standard vs Fixed")
    print("\n   Standard version:")
    print("   - weight = 0.5 → gradient = -0.01 → weight = 0.49")
    print("   - After 50 steps → weight = 0.0 (collapsed!)")
    print("\n   Fixed version:")
    print("   - log_weight = log(0.5) ≈ -0.693")
    print("   - gradient = -0.01 → log_weight = -0.703")
    print("   - weight = exp(-0.703) ≈ 0.495 (still positive!)")
    print("   - After 50 steps → weight > 0 (never collapses!)")

    print("\n" + "=" * 80)
