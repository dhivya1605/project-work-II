"""
tabkanet.py
-----------
TabKANet: KAN-based numerical embedding + Transformer backbone for tabular
crop recommendation.

Architecture:
  1. Each numeric column -> its own FastKAN layer -> token embedding
     (batch, num_features, embed_dim)
  2. A learnable [CLS] token is prepended
  3. Transformer encoder processes the token sequence (self-attention lets
     the model learn interactions between soil/climate features)
  4. The [CLS] token's output representation is passed to an MLP classifier
     head -> logits over the 49 crop classes

This module has no dependency on SFOA/DLO; those are used *around* this
model (SFOA selects which input columns to use, DLO tunes the hyperparameters
below), but TabKANet itself just needs `num_features` and `num_classes`.
"""

import torch
import torch.nn as nn

from kan_layer import NumericalKANEmbedding


class TabKANet(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        embed_dim: int = 32,
        num_grids: int = 8,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        ff_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim

        # 1) KAN-based numeric embedding
        self.kan_embedding = NumericalKANEmbedding(
            num_numeric_features=num_features,
            embed_dim=embed_dim,
            num_grids=num_grids,
        )

        # 2) CLS token + positional embedding (features are unordered but a
        #    learned positional bias still helps the attention distinguish columns)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_features + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # 3) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        # 4) Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, num_classes),
        )

    def forward(self, x_numeric: torch.Tensor) -> torch.Tensor:
        # x_numeric: (batch, num_features) standardized numeric input
        batch_size = x_numeric.shape[0]

        tokens = self.kan_embedding(x_numeric)  # (batch, num_features, embed_dim)
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)
        tokens = torch.cat([cls, tokens], dim=1)          # (batch, num_features+1, embed_dim)
        tokens = tokens + self.pos_embedding

        encoded = self.transformer(tokens)  # (batch, num_features+1, embed_dim)
        cls_out = self.norm(encoded[:, 0, :])  # (batch, embed_dim)

        logits = self.classifier(cls_out)  # (batch, num_classes)
        return logits

    def extract_cls_embedding(self, x_numeric: torch.Tensor) -> torch.Tensor:
        """Useful for SHAP / embedding-space analysis in your dashboard."""
        with torch.no_grad():
            batch_size = x_numeric.shape[0]
            tokens = self.kan_embedding(x_numeric)
            cls = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1) + self.pos_embedding
            encoded = self.transformer(tokens)
            return self.norm(encoded[:, 0, :])
