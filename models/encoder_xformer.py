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
        
        # Use 1D convolution
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
        embed_dim=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        init_std=0.02,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

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
        mask_indices: (B, v) for self-supervised training only
        """
        x = torch.concat(windows, dim=0)        # concat both windows along Batch dimension
        x = self.patch_embed(x)                 # patchify using conv1d
        B, N, D = x.shape                       # N is number of patches per window

        # Add positional embedding
        # pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)   # (1, N, E)
        x = x + self.pos_embed

        # mask w2 - select the visible patches for context encoder (1st half batch)
        if mask_indices is not None:
            #mask_indices = torch.concat(mask_indices, dim=0)  # (2B, m)
            # x = apply_masks(x, mask_indices)
            x, _ = torch.chunk(x, chunks=2, dim=0)      # (B, m, E)

        z = self.transformer(x)                 # (2B, m, E) for target encoder or (B, m, E) for context encoder 

        if self.norm is not None:
            z = self.norm(z)                    # (2B, m, E) or (B, m, E)

        return z


class SeqTransformerEncoder(TransformerEncoder):
    def __init__(
        self,
        seq_length=102,         
        in_channels=6,
        patch_size=17,
        embed_dim=256,
        n_heads=8,
        n_layers=4,
        dropout=0.1,
        init_std=0.02,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()

        # Window modeling for 1D time series
        #self.temporal_conv = MultiScaleTemporalConv(in_channels, 64)

        #self.temporal_norm = nn.LayerNorm(64)

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

        # for modeling window sequence
        self.window_embed = nn.Embedding(2, embed_dim)

        # Self-attention blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
        self.init_std = init_std
        self.apply(self._init_weights)


    def forward(self, windows, mask_indices=None):
        """
        windows: [(B, L, C), (B, L, C)] - batch, sequence length, channels - sequence of windows
        mask_indices: [(B, v), (B, v)] - for self-supervised training only
        """
        B, _, _ = windows[0].shape
        x = torch.concat(windows, dim=0)        # concat both windows along Batch dimension
        
        #x = self.temporal_conv(x)               # modeling the window sequence by capturing different 
        #x = self.temporal_norm(x)
        x = self.patch_embed(x)                 # patchify using conv1d
        # B, N, D = x.shape                       # N is number of patches of the two windows
    
        # Add positional embedding
        # pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)   # (1, N, E)
        x = x + self.pos_embed                   # (2B, N, E)

        # Add window embedding
        window_idx = torch.tensor([0, 1], device=x.device)
        win_emb = self.window_embed(window_idx)   # (2, E)

        win_emb = win_emb.repeat_interleave(B, dim=0) # (2B, E)
        x = x + win_emb.unsqueeze(1)        # (2B, N, E)

        # Select the visible patches for context encoder
        if mask_indices is not None:
            mask_indices = torch.concat(mask_indices, dim=0)    # (2B, m)
            x = apply_masks(x, mask_indices)                    # (2B, m, E)
            # x, _ = torch.chunk(x, chunks=2, dim=0)      # (B, m, E)

        z = self.transformer(x)                 # (2B, m, E) - stack embeddings along Batch dimension

        if self.norm is not None:
            z = self.norm(z)                    # (2B, m, E)

        return z


'''
Design goals (for JEPA + HAR)
For transition modeling, cross-window attention should:
1. Force w2 to attend to w1 (asymmetric)
2. Preserve JEPA’s predictive structure (no leakage from w2 → w1)
3. Be lightweight (HAR data is small)
4. Operate at patch level, not raw samples

w1 → PatchEmbed → Transformer → z1
w2 → PatchEmbed → Transformer → z2

z2 ← CrossAttention(query=z2, key=z1, value=z1)

z1 = self.context_encoder([windows[0]])   # w1 only
z2 = self.target_encoder([windows[1]])    # w2 only

z2_ctx = self.cross_attn(z2, z1)           # transition modeling
z_pred = self.predictor(z1)                # JEPA prediction
z_tgt = z2_ctx.detach()
'''

class CrossWindowAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z_q, z_kv):
        """
        z_q: (B, N2, E)  → w2 patches (queries)
        z_kv: (B, N1, E) → w1 patches (keys/values)
        """
        attn_out, _ = self.attn(
            query=z_q,
            key=z_kv,
            value=z_kv
        )
        return self.norm(z_q + attn_out)



class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        
        '''self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, embed_dim, k, padding=k//2, groups=int(embed_dim/2))
            for k in [3, 7, 15]
        ])'''
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels,
                embed_dim,
                kernel_size=7,
                dilation=d,
                padding=((7 - 1) // 2) * d,
            )
            for d in [1, 2, 4, 8]
        ])

        self.act = nn.GELU()

    def forward(self, x):
        # B, seq_len, E = x.shape
        x = x.transpose(1, 2)  # (B, E, seq_len)

        z = sum(conv(x) for conv in self.convs)

        z = self.act(z)

        z = z.transpose(1, 2) # (B, seq_len, E)
        return z