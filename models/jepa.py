# models/jepa.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import TransformerEncoder
from models.predictor import Predictor
from utils.misc import apply_masks

class JEPA_SEQ(nn.Module):
    def __init__(
        self,
        seq_length,
        channels,
        patch_size,
        num_windows=2,
        embedding_dim=256,
        predictor_embed_dim=512
    ):
        super().__init__()

        self.channels = channels
        self.num_windows = num_windows
        self.embedding_dim = embedding_dim

        # ---- Context Encoder ----
        self.context_encoder = TransformerEncoder(
            seq_length=seq_length, in_channels=channels, patch_size=patch_size
            )
        
        # ---- Target Encoder ----
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad = False
        
        num_patches = self.context_encoder.patch_embed.num_patches
        
        # ---- Predictor ----
        self.predictor = Predictor(
            num_patches=num_patches,
            embed_dim=embedding_dim,
            predictor_embed_dim=predictor_embed_dim
            )

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
          windows: list of k tensors [(B, L, C), (B, L, C)]
          enc_mask_idx: list of k tensors [(B, vi,), (B, vi,)]
          pred_mask_idx: list of k tensors [(B, mi,), (B, mi,)]
        """

        windows = batch["windows"]
        ei = batch["enc_mask_idx"]   # mask_enc
        pi = batch["pred_mask_idx"]    # mask_pred

        # ---- Context encoder (visible patches only) ----
        z_ctx = self.context_encoder(windows, ei)
        
        # ---- Predictor ----
        z_pred = self.predictor(z_ctx, ei, pi)

        # ---- Target encoder (all patches, no mask indices) ----
        with torch.no_grad():
            z = self.target_encoder(windows)
            z = F.layer_norm(z, (z.size(-1),))  # normalize over embedding dimension

            # -- create targets (masked regions of h)
            z_tgt = apply_masks(z, torch.concat(pi, dim=0))

        return z_pred, z_tgt

