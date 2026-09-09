"""
kan_layer.py
------------
A lightweight, efficient Kolmogorov-Arnold Network (KAN) layer used as the
numerical-feature embedding module in TabKANet.

Instead of the original B-spline KAN formulation (which is slow to train),
this uses the common "FastKAN" style approximation: each edge's univariate
activation function is represented as a weighted sum of Gaussian Radial
Basis Functions (RBFs) plus a residual linear (SiLU-gated) term. This is
mathematically a KAN layer (learnable activation per edge, summed at each
node) but is fast and stable enough to train on tabular data.

    phi(x) = sum_i  w_i * exp(-((x - c_i)^2) / (2 * sigma^2))   [spline part]
             + w_base * SiLU(x)                                 [base part]

This layer maps: (batch, in_features) -> (batch, out_features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastKANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_grids: int = 8,
                 grid_min: float = -2.0, grid_max: float = 2.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids

        # Fixed RBF grid centers, one shared grid reused across all edges
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.register_buffer("grid", grid)  # (num_grids,)
        self.sigma = (grid_max - grid_min) / (num_grids - 1)

        # Learnable spline coefficients: one set of RBF weights per (in, out) edge
        self.spline_weight = nn.Parameter(
            torch.randn(in_features, out_features, num_grids) * 0.1
        )
        # Learnable base (residual) weight, like a normal linear layer but SiLU-gated
        self.base_weight = nn.Parameter(
            torch.randn(in_features, out_features) * (1.0 / in_features ** 0.5)
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.layernorm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features)
        batch = x.shape[0]

        # ---- base (residual) term ----
        base = F.silu(x) @ self.base_weight  # (batch, out_features)

        # ---- spline (RBF) term ----
        # x_expanded: (batch, in_features, 1) vs grid (num_grids,)
        x_expanded = x.unsqueeze(-1)  # (batch, in, 1)
        rbf = torch.exp(-((x_expanded - self.grid) ** 2) / (2 * self.sigma ** 2))
        # rbf: (batch, in_features, num_grids)

        # Combine per-edge RBF activations with per-edge spline weights, then
        # sum over input features to get each output node's value.
        # einsum: batch,in,g  *  in,out,g  ->  batch,out
        spline = torch.einsum("big,iog->bo", rbf, self.spline_weight)

        out = base + spline + self.bias
        return self.layernorm(out)


class NumericalKANEmbedding(nn.Module):
    """
    Embeds each numerical column independently through its own small FastKAN
    layer, producing a (batch, num_numeric_features, embed_dim) tensor that
    can be fed into a Transformer encoder (this is the "TabKANet" idea:
    KAN-based numerical embeddings + Transformer backbone).
    """

    def __init__(self, num_numeric_features: int, embed_dim: int, num_grids: int = 8):
        super().__init__()
        self.num_numeric_features = num_numeric_features
        self.embed_dim = embed_dim
        # One independent FastKAN layer (1 -> embed_dim) per numeric column
        self.column_kans = nn.ModuleList([
            FastKANLayer(in_features=1, out_features=embed_dim, num_grids=num_grids)
            for _ in range(num_numeric_features)
        ])

    def forward(self, x_numeric: torch.Tensor) -> torch.Tensor:
        # x_numeric: (batch, num_numeric_features), already standardized
        outs = []
        for i, kan in enumerate(self.column_kans):
            col = x_numeric[:, i:i + 1]  # (batch, 1)
            outs.append(kan(col))        # (batch, embed_dim)
        return torch.stack(outs, dim=1)   # (batch, num_numeric_features, embed_dim)
