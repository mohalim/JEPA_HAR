# models/positional_transformer_encoder.py
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))
    
    def get(self, T):
        return self.pos_embed[:T]  # (T, E)
    '''
    def forward(self, x):
        """
        x: [B, seq_len, E]
        """
        B, L, E = x.shape
        # Add positional encoding to all sequence positions
        # pos_embed = self.pos_embed[:, :, :].expand(B, -1, -1)  # (B, L, E)
        pos_embed = self.pos_embed[:, :L, :]  # (B, L, E)
        return x + pos_embed'''

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,      # input dim, window's number of channel
        seq_len,        # sequence length
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

        # window / segment embedding
        self.window_embed = nn.Embedding(num_windows, embed_dim)

        # channel embedding
        self.channel_embed = nn.Embedding(input_dim, embed_dim)
        
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
    
    def forward(self, x1, x2, mask):
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

        # Window embeddings
        w1_id = torch.zeros(seq_len, dtype=torch.long, device=x1.device)
        w2_id = torch.ones(seq_len, dtype=torch.long, device=x2.device)

        w1_embed = self.window_embed(w1_id)[None, :, :]  # (1, seq_len, E)
        w2_embed = self.window_embed(w2_id)[None, :, :]  # (1, seq_len, E)

        # ---- Positional embeddings ----
        pos = self.pos_enc.get(seq_len)
        
        # ---- Channel embeddings ----
        ch_ids = torch.arange(C, device=x1.device)      # (C,)
        ch_embed = self.channel_embed(ch_ids)           # (C, E)
        ch_embed = ch_embed.unsqueeze(0).unsqueeze(0)   # (1, 1, C, E)
        ch_embed = ch_embed.expand(B, seq_len, -1, -1)  # (B, seq_len, C, E)

        # Sum channel embeddings along channel dimension
        # Add to projected embeddings directly (after projection)
        x1 = x1 + pos + w1_embed + ch_embed.sum(dim=2)   # Add positional encoding
        x2 = x2 + pos + w2_embed + ch_embed.sum(dim=2)

        x = torch.cat([x1, x2], dim=1)  # (B, 2seq_len, E)
        
        z = self.encoder(x)         # (B, 2seq_len, E)
        # Learned temporal filters (frequency modeling)
        z = self.temporal_conv(z)             # (B, seq_len+seq_len, E)
        return z


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, embed_dim, num_windows):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, k, padding=k//2, groups=int(embed_dim/num_windows))
            for k in [3, 7, 15]
        ])
        # self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # B, seq_len, E = x.shape
        x = x.transpose(1, 2)  # (B, E, seq_len)

        z = sum(conv(x) for conv in self.convs)
        # z = self.norm(z)
        z = self.act(z)

        z = z.transpose(1, 2) # (B, seq_len, E)
        return z