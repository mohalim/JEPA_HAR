import torch
import torch.nn as nn

from utils.misc import trunc_normal_, apply_masks

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

class ConvBlock(nn.Module):
    def __init__(self, dim, kernel_size, drop_path=0.):
        super().__init__()
        # Depthwise convolution to capture temporal dependencies
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        # Pointwise expansions (similar to FeedForward in Transformer)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # self.drop_path = nn.Identity() # Add Stochastic Depth if needed

    def forward(self, x):
        """ x: (B, N, E) where N is num_patches """
        input = x
        # 1. Temporal modeling
        x = x.transpose(1, 2)   # (B, E, N)
        x = self.dwconv(x)
        x = x.transpose(1, 2)   # (B, N, E)
        x = self.norm(x)
        
        # 2. Channel modeling (MLP)
        x = self.pwconv1(x)     # (B, N, 4E)
        x = self.act(x)         # (B, N, 4E)
        x = self.pwconv2(x)     # (B, N, E)
        
        return input + x

class ConvolutionalEncoder(nn.Module):

    def __init__(
        self,
        seq_length=102,         
        in_channels=6,
        patch_size=17,
        kernel_sizes=[7, 5, 3, 3],
        embed_dim=256,
        # n_layers=4,
        init_std=0.02,
        norm_layer=nn.LayerNorm
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

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # for modeling window sequence
        self.window_embed = nn.Embedding(2, embed_dim)
        
        # Stack of convolutional blocks
        self.blocks = nn.ModuleList([
            ConvBlock(embed_dim, k_size) for k_size in kernel_sizes
        ])
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

    def forward(self, windows, mask_indices=None):
        """
        windows: [(B, L, C), (B, L, C)] - batch, sequence length, channels - sequence of windows
        mask_indices: (B, v) for self-supervised training only
        """
        B, _, _ = windows[0].shape
        # 1. Combine windows
        x = torch.concat(windows, dim=0) 
        x = self.patch_embed(x)
        
        # 2. Add Positional Info
        x = x + self.pos_embed

        # Add window embedding
        window_idx = torch.tensor([0, 1], device=x.device)
        win_emb = self.window_embed(window_idx)   # (2, E)

        win_emb = win_emb.repeat_interleave(B, dim=0) # (2B, E)
        x = x + win_emb.unsqueeze(1)        # (2B, N, E)
        
        # 3. Apply Masking (Crucial for JEPA)
        # In a Conv encoder, we treat patches as a sequence. 
        # If mask_indices is provided, we only keep the visible patches.
        if mask_indices is not None:
            mask_indices = torch.concat(mask_indices, dim=0)    # (2B, m)
            # If mask_indices is (B, m), we gather the visible patches
            # This turns the 'image' into a 'sequence' of patches
            x = apply_masks(x, mask_indices) 

        # 4. Pass through Conv Blocks
        for block in self.blocks:
            x = block(x)
            
        if self.norm is not None:
            return self.norm(x)                    # (2B, m, E)
        else:
            return x

        
