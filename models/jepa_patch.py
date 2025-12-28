# models/jepa.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.patch_transformer_encoder import PatchTransformerEncoder, 
from models.predictor import Predictor

class JEPA(nn.Module):
    def __init__(self, patch_dim, embedding_dim, num_patches=5, hidden_dim=512):
        super().__init__()

        self.num_patches = num_patches
        
        self.context_encoder = PatchTransformerEncoder(
            patch_dim, embedding_dim, num_patches
        )
        self.target_encoder = PatchTransformerEncoder(
            patch_dim, embedding_dim, num_patches
        )
        self.predictor = Predictor(embedding_dim, hidden_dim)

        # Initialize target encoder
        for ctx_param, tgt_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            tgt_param.data.copy_(ctx_param.data)
            tgt_param.requires_grad = False

    @torch.no_grad()
    def update_target(self, momentum=0.99):
        for ctx_param, tgt_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            tgt_param.data = momentum * tgt_param.data + (1 - momentum) * ctx_param.data

    def forward(self, batch):
        """
        batch keys:
          visible_patches: [B, Nv, D]
          masked_patches:  [B, Nm, D]
          visible_idx:     [B, Nv]
          masked_idx:      [B, Nm]
          all_patches:     [B, N, D]
        """

        B = batch["visible_patches"].size(0)

        # ----- Context encoder (visible only) -----
        z_ctx = self.context_encoder(
            batch["visible_patches"],
            batch["visible_idx"]   # same indices for whole batch
        )                             # [B, Nv, E]

        # Aggregate context tokens
        z_ctx = z_ctx.mean(dim=1)     # [B, E]

        z_pred = self.predictor(z_ctx) # [B, E]

        # ----- Target encoder (full, no mask) -----
        with torch.no_grad():
            z_all = self.target_encoder(
                batch["all_patches"],
                torch.arange(batch["all_patches"].size(1), device=z_ctx.device)
            )                         # [B, N, E]

            z_tgt = z_all[:, batch["masked_idx"][0], :].mean(dim=1)

        return z_pred, z_tgt


class JEPA_SEQ(nn.Module):
    def __init__(
        self,
        patch_dim,
        embedding_dim,
        channels,
        # num_patches=5,
        num_windows=2,
        hidden_dim=512
    ):
        super().__init__()

        self.channels = channels
        self.num_windows = num_windows

        self.temporal_conv = MultiScaleTemporalConv(channels)
        self.repr_proj = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256)
        )


        # ---- Encoders ----
        self.context_encoder = PatchTransformerEncoder(
            patch_dim, embedding_dim, num_patches * num_windows
        )

        self.target_encoder = PatchTransformerEncoder(
            patch_dim, embedding_dim, num_patches * num_windows
        )

        pred_input_dim = int(embedding_dim / channels * embedding_dim / channels * 2)
        self.predictor = Predictor(pred_input_dim, hidden_dim)

        # projection for cosine similarity
        self.cosine_proj = nn.Linear(channels, channels, bias=False)

        # Initialize target encoder
        for ctx_param, tgt_param in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            tgt_param.data.copy_(ctx_param.data)
            tgt_param.requires_grad = False

    @torch.no_grad()
    def update_target(self, momentum=0.99):
        for ctx_param, tgt_param in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            tgt_param.data = momentum * tgt_param.data + (1 - momentum) * ctx_param.data
            

    def temporal_encode(self, z):
        """
        z: (B, N, E)
        """
        B, N, E = z.shape
        t = E // self.channels

        # Reshape into temporal structure
        z = z.view(B, N, t, self.channels)   # (B,N,T,C)

        # Learned temporal filters (frequency modeling)
        z = self.temporal_conv(z)             # (B,N,T,C)

        # Temporal pooling (invariant summary)
        z = z.mean(dim=2)                     # (B,N,C)

        # Flatten patches
        z = z.flatten(start_dim=1)            # (B, N*C)

        # Compact projection
        z = self.repr_proj(z)                 # (B,256)
        return z


    def forward(self, batch):
        """
        batch keys:
          visible_patches: list of k tensors [(B, Nv1, E), (B, Nv2, E)]
          visible_idx:     list of k tensors [(B, Nv1,), (B, Nv2,)]
          masked_idx:      list of k tensors [(B, Nm1,), (B, Nm2,)]
          all_patches:     list of k tensors [(B, N, E), (B, N, E)]
        """

        # ---- Context encoder (visible only, concatenated across time) ----
        v1, v2 = batch["visible_patches"]
        idx1, idx2 = batch["visible_idx"]
        
        len_v = v1.size(1)
        B, Nv, E = v1.shape

        # concat patches across sequence dimension
        x = torch.cat([v1, v2], dim=1)          # (B, Nv1+Nv2, D)
        x_idx = torch.cat([idx1, idx2], dim=1)  # (B, Nv1+Nv2)

        # add batch dim
        z = self.context_encoder(x, x_idx)      # (B, Nv1+Nv2, E)

        z_ctx = self.temporal_encode(z)

        # ---- Target encoder (masked prediction) ----
        with torch.no_grad():
            w1, w2 = batch["all_patches"]
            idx_m1, idx_m2 = batch["masked_idx"]
            idx_m = idx_m1

            len_w = w1.size(1)
            w = torch.cat([w1, w2], dim=1)          # (B, N1+N2, D)

            B, N, _ = w1.shape
            device = w1.device

            idx_w1 = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
            idx_w2 = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1) + N

            w_idx = torch.cat([idx_w1, idx_w2], dim=1)  # (B, 2N)

            z_all = self.target_encoder(w, w_idx)       # (B, 2N)

            z_all = self.temporal_encode(z_all)

            # mask AFTER encoding but BEFORE pooling across windows
            idx_exp = idx_m.unsqueeze(-1).expand(-1, -1, z_all.size(-1))
            z_tgt = torch.gather(z_all, dim=1, index=idx_exp).mean(dim=1)

        # ---- Predictor head ----
        z_pred = self.predictor(z_ctx)

        return z_pred, z_tgt
    

class MultiScaleTemporalConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, k, padding=k//2, groups=channels)
            for k in [3, 7, 15]
        ])
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        B, N, T, C = x.shape
        x = x.reshape(B*N, T, C).transpose(1, 2)  # (BN, C, T)

        y = sum(conv(x) for conv in self.convs)
        y = self.norm(y)
        y = self.act(y)

        y = y.transpose(1, 2).reshape(B, N, T, C)
        return y

