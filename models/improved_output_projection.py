#!/usr/bin/env python3
"""
Improved Output Projection Layer

Addresses the issue of overfitting caused by large dimension jump (56 → 1536).
Provides multiple strategies for better regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressiveProjection(nn.Module):
    """
    Gradual dimension expansion with regularization.

    56 → 128 → 256 → 512 → 1536

    Benefits:
    - Smoother information flow
    - Better gradient propagation
    - Natural regularization
    - Easier optimization
    """

    def __init__(
        self,
        input_dim=56,
        output_dim=1536,
        intermediate_dims=[128, 256, 512],
        dropout=0.2,
        use_layer_norm=True,
        use_residual=False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual

        # Build progressive layers
        dims = [input_dim] + intermediate_dims + [output_dim]
        self.layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]

            layer_components = []

            # Linear
            layer_components.append(nn.Linear(in_d, out_d))

            # Normalization (except last layer)
            if i < len(dims) - 2 and use_layer_norm:
                layer_components.append(nn.LayerNorm(out_d))

            # Activation (except last layer)
            if i < len(dims) - 2:
                layer_components.append(nn.SiLU())

                # Dropout
                if dropout > 0:
                    layer_components.append(nn.Dropout(dropout))

            self.layers.append(nn.Sequential(*layer_components))

        # Final normalization
        if use_layer_norm:
            self.final_norm = nn.LayerNorm(output_dim)
        else:
            self.final_norm = None

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]

        Returns:
            [batch_size, output_dim]
        """
        for layer in self.layers:
            x = layer(x)

        if self.final_norm is not None:
            x = self.final_norm(x)

        return x


class BottleneckProjection(nn.Module):
    """
    Bottleneck architecture with dimension reduction then expansion.

    56 → 32 (bottleneck) → 1536

    Benefits:
    - Forces compression of information
    - Prevents memorization
    - Acts as regularization
    """

    def __init__(
        self,
        input_dim=56,
        output_dim=1536,
        bottleneck_dim=32,
        hidden_dim=256,
        dropout=0.2,
        use_layer_norm=True
    ):
        super().__init__()

        # Encoder: compress to bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        # Decoder: expand to output
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim]

        Returns:
            [batch_size, output_dim]
        """
        # Compress
        bottleneck = self.encoder(x)

        # Expand
        output = self.decoder(bottleneck)

        return output


class AdaptiveProjection(nn.Module):
    """
    Adaptive projection that matches target dimensionality.

    If target is high-dim (e.g., 1536), uses progressive projection.
    If target is low-dim (e.g., 512), uses simple projection.

    Also supports dimension adapter for mismatched ligand embeddings.
    """

    def __init__(
        self,
        input_dim=56,
        output_dim=1536,
        dropout=0.2,
        use_layer_norm=True,
        strategy='auto'  # 'auto', 'progressive', 'bottleneck', 'simple'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Auto-select strategy
        if strategy == 'auto':
            dim_ratio = output_dim / input_dim
            if dim_ratio > 20:
                strategy = 'progressive'
            elif dim_ratio > 5:
                strategy = 'bottleneck'
            else:
                strategy = 'simple'

        print(f"Using {strategy} projection: {input_dim} → {output_dim}")

        # Build projection based on strategy
        if strategy == 'progressive':
            # Calculate intermediate dimensions
            num_steps = max(3, int(torch.log2(torch.tensor(dim_ratio)).item()))
            intermediate_dims = []
            current = input_dim
            step_ratio = (output_dim / input_dim) ** (1 / (num_steps + 1))

            for _ in range(num_steps):
                current = int(current * step_ratio)
                intermediate_dims.append(current)

            self.projection = ProgressiveProjection(
                input_dim=input_dim,
                output_dim=output_dim,
                intermediate_dims=intermediate_dims,
                dropout=dropout,
                use_layer_norm=use_layer_norm
            )

        elif strategy == 'bottleneck':
            bottleneck_dim = max(32, input_dim // 2)
            hidden_dim = int((input_dim + output_dim) / 3)

            self.projection = BottleneckProjection(
                input_dim=input_dim,
                output_dim=output_dim,
                bottleneck_dim=bottleneck_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_layer_norm=use_layer_norm
            )

        else:  # simple
            self.projection = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()
            )

    def forward(self, x):
        return self.projection(x)


# Example usage and comparison
if __name__ == "__main__":
    # Test different projections
    batch_size = 4
    input_dim = 56
    output_dim = 1536

    x = torch.randn(batch_size, input_dim)

    print("="*80)
    print("Testing Output Projection Strategies")
    print("="*80)

    # 1. Simple (current approach)
    simple = nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim)
    )
    y_simple = simple(x)
    params_simple = sum(p.numel() for p in simple.parameters())
    print(f"\n1. Simple: {input_dim} → {output_dim}")
    print(f"   Parameters: {params_simple:,}")
    print(f"   Output shape: {y_simple.shape}")

    # 2. Progressive
    progressive = ProgressiveProjection(
        input_dim=input_dim,
        output_dim=output_dim,
        intermediate_dims=[128, 256, 512],
        dropout=0.2
    )
    y_prog = progressive(x)
    params_prog = sum(p.numel() for p in progressive.parameters())
    print(f"\n2. Progressive: {input_dim} → 128 → 256 → 512 → {output_dim}")
    print(f"   Parameters: {params_prog:,}")
    print(f"   Output shape: {y_prog.shape}")
    print(f"   Param reduction: {(1 - params_prog/params_simple)*100:.1f}%")

    # 3. Bottleneck
    bottleneck = BottleneckProjection(
        input_dim=input_dim,
        output_dim=output_dim,
        bottleneck_dim=32,
        hidden_dim=256,
        dropout=0.2
    )
    y_bottle = bottleneck(x)
    params_bottle = sum(p.numel() for p in bottleneck.parameters())
    print(f"\n3. Bottleneck: {input_dim} → 32 → 256 → {output_dim}")
    print(f"   Parameters: {params_bottle:,}")
    print(f"   Output shape: {y_bottle.shape}")
    print(f"   Param reduction: {(1 - params_bottle/params_simple)*100:.1f}%")

    # 4. Adaptive
    adaptive = AdaptiveProjection(
        input_dim=input_dim,
        output_dim=output_dim,
        dropout=0.2,
        strategy='auto'
    )
    y_adaptive = adaptive(x)
    params_adaptive = sum(p.numel() for p in adaptive.parameters())
    print(f"\n4. Adaptive (auto-selected)")
    print(f"   Parameters: {params_adaptive:,}")
    print(f"   Output shape: {y_adaptive.shape}")

    print("\n" + "="*80)
    print("Recommendation for 56 → 1536:")
    print("- Progressive: Better gradient flow, more parameters but better regularization")
    print("- Bottleneck: Strong regularization, prevents overfitting")
    print("- Adaptive: Automatically selects best strategy")
    print("="*80)
