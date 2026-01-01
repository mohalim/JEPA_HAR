# models/encoder.py
# Reference: https://github.com/facebookresearch/ijepa/tree/main
import torch
import torch.nn as nn

from utils.misc import trunc_normal_, apply_masks, get_1d_sincos_pos_embed

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

class PatchEmbed1D(nn.Module):
    """1D Signal to Patch Embedding for time series data"""
    def __init__(self, seq_length=128, patch_size=16, in_channels=6, embed_dim=768):
        super().__init__()
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = seq_length // patch_size
        
        # Use 1D convolution instead of 2D
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        x: (B, L, C) where B is batch size, L is sequence length, C is channels
        output: (B, num_patches, embed_dim)
        """
        B, L, C = x.shape
        # assert L == self.seq_length, f"Input sequence length ({L}) doesn't match model ({self.seq_length})"
        x = x.transpose(1, 2)   # (B, channels, seq_len)
        x = self.proj(x)        # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)   # (B, num_patches, embed_dim)
        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        seq_length=102,
        in_channels=6,
        patch_size=17,
        embed_dim=256,
        # num_windows=2,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        init_std=0.02,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        # self.num_windows = num_windows

        # Patch embedding for 1D time series
        self.patch_embed = PatchEmbed1D(
            seq_length=seq_length,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            num_patches,
            cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        self.norm = norm_layer(embed_dim)
        self.init_std = init_std
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def interpolate_pos_encoding(self, x, pos_embed):
        """Interpolate positional encoding for different sequence lengths"""
        npatch = x.shape[1]
        N = pos_embed.shape[1]
        if npatch == N:
            return pos_embed
        
        # Linear interpolation for 1D sequences
        pos_embed = pos_embed.permute(0, 2, 1)  # (1, D, N)
        pos_embed = nn.functional.interpolate(
            pos_embed,
            size=npatch,
            mode='linear',
            align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 1)  # (1, npatch, D)
        return pos_embed

    def forward(self, windows, mask_indices=None):
        """
        windows: [(B, L, C), (B, L, C)] - batch, sequence length, channels - sequence of windows
        mask_indices: (B, mi,) - batch, visible indices - optional for self-supervised training only
        """
        x = torch.concat(windows, dim=0)        # concat both windows along Batch dimension
        x = self.patch_embed(x)                 # patchify using conv1d
        B, N, D = x.shape                       # N is number of patches per window

        #x1, x2 = torch.chunk(x, chunks=2, dim=0)    # split back the window patches

        # Add positional embedding
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)   # (1, N, E)
        x = x + pos_embed
        #x1 = x1 + pos_embed
        #x2 = x2 + pos_embed

        # mask x - select the visible patches for context encoder
        if mask_indices is not None:
            mask_indices = torch.concat(mask_indices, dim=0)  # (2B, m)
            x = apply_masks(x, mask_indices)

        z = self.transformer(x)                 # (2B, m, E) - stack embeddings along Batch dimension

        if self.norm is not None:
            z = self.norm(z)                    # (2B, m, E)

        return z



