# models/predictor.py
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_heads=2,
        n_layers=1,
        dropout=0.1
    ):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(layer, n_layers)

        self.head = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: (B, T, E)
        """
        z = self.transformer(x)

        return self.head(z)

