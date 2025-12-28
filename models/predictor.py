# models/predictor.py
import torch.nn as nn
from models.pos_transformer_encoder import PositionalEncoding


class Predictor(nn.Module):
    def __init__(
        self,
        embed_dim,
        seq_len,
        n_heads=4,
        n_layers=1,
        dropout=0.1
    ):
        super().__init__()

        self.pos_enc = PositionalEncoding(seq_len * 2, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(layer, n_layers)

        self.head = nn.Linear(embed_dim, embed_dim)

    def forward(self, z_ctx_tokens):
        """
        z_ctx_tokens: (B, 2T, E)
        """
        z = self.pos_enc(z_ctx_tokens)
        z = self.transformer(z)

        T = z.shape[1] // 2
        z_pred = z[:, T:, :]   # predict only w2

        return self.head(z_pred)

