# models/jepa_xformer.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder_xformer import TransformerEncoder, SeqTransformerEncoder
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
        n_heads=8,
        n_layers=4,
        predictor_embed_dim=128,
        predictor_n_heads=2,
        predictor_n_layers=1,
        is_seq=False
    ):
        super().__init__()

        self.channels = channels
        self.num_windows = num_windows
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim

        # ---- Context Encoder ----
        if not is_seq:
            self.context_encoder = TransformerEncoder(
                seq_length=seq_length, in_channels=channels, patch_size=patch_size,
                embed_dim=embedding_dim, n_heads=n_heads, n_layers=n_layers, norm_layer=nn.LayerNorm
                )
        else:
            self.context_encoder = SeqTransformerEncoder(
                seq_length=seq_length, in_channels=channels, patch_size=patch_size,
                embed_dim=embedding_dim, n_heads=n_heads, n_layers=n_layers, norm_layer=nn.LayerNorm
                )
        
        # ---- Target Encoder ----
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad = False
        
        num_patches = self.context_encoder.patch_embed.num_patches
        
        # ---- Predictor ----
        self.predictor = Predictor(
            num_patches=num_patches,
            embed_dim=embedding_dim,
            predictor_embed_dim=predictor_embed_dim,
            predictor_n_heads=predictor_n_heads,
            predictor_n_layers=predictor_n_layers,
            )

    @torch.no_grad()
    def update_target(self, momentum=0.99):
        for ctx_param, tgt_param in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            # tgt_param.data = momentum * tgt_param.data + (1 - momentum) * ctx_param.data
            tgt_param.data.mul_(momentum).add_((1.-momentum) * ctx_param.detach().data)

    def forward(self, batch):
        """
        batch keys:
          windows: list of k tensors [(B, L, C), (B, L, C)]
          enc_mask_idx: list of k indices [(B, vi,), (B, vi)]
          pred_mask_idx: list of k indices [(B, mi,), (B, mi)]
        """
        windows = batch["windows"]
        ei = batch["enc_mask_idx"]   # mask_enc
        pi = batch["pred_mask_idx"]    # mask_pred
        # B, _, _ = windows[0].shape

        # ---- Context encoder (visible patches only) ----
        z_ctx = self.context_encoder(windows, ei)
        
        # ---- Predictor ----
        z_pred = self.predictor(z_ctx, ei, pi)

        # ---- Target encoder (all patches, no mask indices) ----
        with torch.no_grad():
            z = self.target_encoder(windows)    # (2B, N, E) 
            z = F.layer_norm(z, (z.size(-1),))  # normalize over embedding dimension

            # -- create targets (masked regions of h)
            z_tgt = apply_masks(z, torch.concat(pi, dim=0))
            # z_tgt = z[B:]

        return z_pred, z_tgt, z_ctx