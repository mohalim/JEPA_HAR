# models/jepa.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pos_transformer_encoder import TransformerEncoder
from models.predictor import Predictor


class JEPA_SEQ(nn.Module):
    def __init__(
        self,
        input_channels,
        embedding_dim,
        window_size,
        num_windows=2,
        hidden_dim=512
    ):
        super().__init__()

        self.channels = input_channels
        self.num_windows = num_windows

        # ---- Encoders ----
        self.context_encoder = TransformerEncoder(
            input_channels, window_size, embedding_dim, num_windows
        )

        self.target_encoder = TransformerEncoder(
            input_channels, window_size, embedding_dim, num_windows
        )

        self.predictor = Predictor(embedding_dim)

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

    def forward(self, batch):
        """
        batch keys:
          w1: (B, seq_len, C)
          w2:  (B, seq_len, C)
          masked_w2: (B, seq_len, C) - some samples are zeroed
          mask: (B, seq_len) - boolean
        """

        # ---- Context encoder (visible only, concatenated across time) ----
        w1 = batch["w1"]
        w2m = batch["masked_w2"]
        mask = batch["mask"]
  
        B, T, _ = w1.shape

        T = w1.size(1)

        z = self.context_encoder(w1, w2m, mask)      # (B, seq_len+seq_len, E)

        z_ctx = z[:, T:, :]  # (B, seq_len, E)

        # ---- Target encoder (masked prediction) ----
        with torch.no_grad():
            w2 = batch["w2"]
            full_mask = torch.ones_like(mask, dtype=torch.bool)
            z = self.target_encoder(w1, w2, mask=full_mask) # (B, seq_len+seq_len, E)

            # select only window 2 tokens
            z_tgt = z[:, T:, :]  # (B, seq_len, E)

        # ---------------- Select MASKED positions ONLY ----------------
        masked_idx = ~mask                          # False → masked (B, T)  
        masked_idx_exp = masked_idx.unsqueeze(-1)   # (B, T, 1)

        z_ctx_masked = z_ctx.masked_select(masked_idx_exp)
        z_tgt_masked = z_tgt.masked_select(masked_idx_exp)

        E = z_ctx.size(-1)
        N_masked = masked_idx.sum(dim=1)[0].item()

        z_ctx_masked = z_ctx_masked.view(B, N_masked, E)
        z_tgt_masked = z_tgt_masked.view(B, N_masked, E)

        # ---- Predictor ----
        z_pred = self.predictor(z_ctx_masked)

        return z_pred, z_tgt_masked