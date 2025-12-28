import torch
import torch.nn as nn

class ConvolutionalEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        c_out: list,                 # e.g. [64, 128, 256]
        kernel_size: int = 5,
        embed_dim: int = None,
        conv_bias = True
    ):
        super().__init__()

        assert len(c_out) > 0, "c_out must contain at least one layer"
        self.embed_dim = embed_dim or c_out[-1]

        self.stages = nn.ModuleList()
        c_in = in_channels

        for c in c_out:
            stage = nn.Sequential(
                # Conv 1
                nn.Conv1d(c_in, c, kernel_size, padding=kernel_size // 2, bias=conv_bias),
                nn.BatchNorm1d(c),
                nn.GELU(),

                # Conv 2
                nn.Conv1d(c, c, kernel_size, padding=kernel_size // 2, bias=conv_bias),
                nn.BatchNorm1d(c),
                nn.GELU(),

                # Downsample Conv
                nn.Conv1d(c, c, kernel_size, stride=2, padding=kernel_size // 2, bias=conv_bias),
                nn.BatchNorm1d(c),
                nn.GELU(),
            )
            self.stages.append(stage)
            c_in = c

        '''
        self.proj = nn.Sequential(
            nn.Linear(c_out[-1], self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        )'''

    def forward(self, x, return_embeddings=True):
        """
        x:
          (B, C, T)        or
          (B, N, C, t)

        return:
          feature_maps: list of tensors [(B, C_i, T_i), ...]
          embeddings:   list of tensors [(B, D_i), ...] (optional)
        """

        # Handle patchified input
        if x.dim() == 4:
            B, N, C, t = x.shape
            x = x.view(B * N, C, t)
            is_patchified = True
        else:
            B = x.shape[0]
            N = None
            is_patchified = False

        feature_maps = []
        embeddings = []

        for i, stage in enumerate(self.stages):
            x = stage(x)                   # (B*N, C_i, T_i)
            feature_maps.append(x)

            if return_embeddings:
                z = x.mean(dim=-1)         # global temporal pooling
                if i == len(self.stages) - 1:
                    z = self.proj(z)       # project only final stage
                embeddings.append(z)

        # Reshape back to (B, N, ...)
        if is_patchified:
            feature_maps = [
                fm.view(B, N, fm.size(1), fm.size(2))
                for fm in feature_maps
            ]
            embeddings = [
                emb.view(B, N, -1)
                for emb in embeddings
            ]

        return feature_maps, embeddings
