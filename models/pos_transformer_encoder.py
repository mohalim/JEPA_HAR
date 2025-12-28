# models/positional_transformer_encoder.py
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.randn(1, seq_len, embed_dim)
        )
    
    def forward(self, x):
        """
        x: [B, seq_len, E]
        """
        B, L, E = x.shape
        # Add positional encoding to all sequence positions
        pos_embed = self.pos_embed[:, :L, :].expand(B, -1, -1)  # (B, L, E)
        return x + pos_embed


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,      # input dim = w1 channel dim + w2 channel dim
        seq_len,        # expected sequence length
        embed_dim=128,
        num_windows=2,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.proj = nn.Linear(input_dim, embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Initialize it with a small standard deviation
        nn.init.normal_(self.mask_token, std=.02)

        self.pos_enc = PositionalEncoding(seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.temporal_conv = MultiScaleTemporalConv(embed_dim, num_windows)

    def temporal_encode(self, x):
        """
        x: (B, seq_len, E)
        """
        # Learned temporal filters (frequency modeling)
        z = self.temporal_conv(x)             # (B, seq_len+seq_len, E)

        # Temporal pooling (invariant summary)
        z = z.mean(dim=1)                     # (B, E)

        return z
    
    def forward(self, x1, x2, mask, return_summary=False):
        """
        x1: (B, seq_len, channels)
        x2: (B, seq_len, channels)
        mask: (B, seq_len)
        """
        B, seq_len, C = x1.shape
        x1 = self.proj(x1)            # (B, seq_len, E)
        x2 = self.proj(x2)            # (B, seq_len, E)

        # ---- Apply learned mask token to x2 embeddings ----
        # mask: True = keep, False = mask
        mask = mask.unsqueeze(-1)       # (B, seq_len, 1)
        x2 = torch.where(
            mask,
            x2,
            self.mask_token.expand(B, seq_len, -1)
        )

        x = torch.cat([x1, x2], dim=1)  # (B, 2seq_len, E)

        x = self.pos_enc(x)         # Add positional encoding
        z = self.encoder(x)         # (B, 2seq_len, E)
        # Learned temporal filters (frequency modeling)
        z =  self.temporal_conv(z)             # (B, seq_len+seq_len, E)

        return z


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, embed_dim, num_windows):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, k, padding=k//2, groups=int(embed_dim/num_windows))
            for k in [3, 7, 15]
        ])
        self.norm = nn.BatchNorm1d(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # B, seq_len, E = x.shape
        x = x.transpose(1, 2)  # (B, E, seq_len)

        z = sum(conv(x) for conv in self.convs)
        z = self.norm(z)
        z = self.act(z)

        z = z.transpose(1, 2) # (B, seq_len, E)
        return z