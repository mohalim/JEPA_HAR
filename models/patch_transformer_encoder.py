# models/patch_transformer_encoder.py
import torch
import torch.nn as nn
import math

# Patch positional encoder doesn't work like Sinusoidal positional encoder
# which encodes based on the patch sequence.

# In patch positional encoder, temporal order is not preserved 
# by the order of tokens in the input tensor but by the positional
# encodings added to each token.
# Patches (temporal subwindows) may be randomly permuted or subsampled,
# but each patch is always paired with its correct absolute positional
# embedding via its patch_indices.

# Why this method works with self-attention? The patches are injected with 
# positional information which lives in the patch indices (not in the tensor order).
# Once the information is injected, self-attention is permutation-invariant, which
# means the Transformer knows a patch belongs to which time step.

class PatchPositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, embed_dim)
        )

    def forward(self, x, patch_indices):
        """
        x: [B, N, E]
        patch_indices: [N]
        """
        B, N, E = x.shape
        # pos = self.pos_embed[:, patch_indices, :]

        # Expand pos_embed to batch
        pos_embed = self.pos_embed.expand(B, -1, -1)   # [B, num_patches, E]
        pos = torch.gather(
            pos_embed,
            dim=1,
            index=patch_indices.unsqueeze(-1).expand(-1, -1, E)
        )                                               # [B, N, E]
        return x + pos


class PatchTransformerEncoder(nn.Module):
    def __init__(
        self,
        patch_dim,
        embed_dim=128,
        num_patches=5,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.proj = nn.Linear(patch_dim, embed_dim)
        self.pos_enc = PatchPositionalEncoding(num_patches, embed_dim)

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

    def forward(self, patches, patch_indices):
        """
        patches: [B, N, D]
        patch_indices: [N]
        """
        x = self.proj(patches)                  # [B, N, E]
        x = self.pos_enc(x, patch_indices)
        x = self.encoder(x)
        return x                                # [B, N, E]


