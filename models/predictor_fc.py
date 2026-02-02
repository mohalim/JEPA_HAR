# models/predictor.py
import torch
import torch.nn as nn

from utils.misc import trunc_normal_, get_1d_sincos_pos_embed, apply_masks

class Predictor(nn.Module):
    def __init__(
        self,
        num_patches=17,
        embed_dim=64,
        predictor_embed_dim=32,
        dropout=0.3,
        init_std=0.02,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        # Map encoder embeddings → predictor space
        #self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        self.ctx_pool = nn.Sequential(
            nn.Linear(embed_dim, predictor_embed_dim),
            nn.LayerNorm(predictor_embed_dim)
        )

        # Mask token
        #self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # Fixed sinusoidal positional embedding (JEPA-consistent)
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim),
            requires_grad=True
        )
        #predictor_pos_embed = get_1d_sincos_pos_embed(
        #    predictor_embed_dim,
        #    num_patches,
        #    cls_token=False
        #)
        #self.predictor_pos_embed.data.copy_(
        #    torch.from_numpy(predictor_pos_embed).float().unsqueeze(0)
        #)

        # ---- Lightweight JEPA Predictor ----
        self.predictor_mlp = nn.Sequential(
            norm_layer(predictor_embed_dim),
            nn.Linear(predictor_embed_dim, predictor_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # nn.Linear(predictor_embed_dim, predictor_embed_dim),
        )

        # Project back to encoder embedding space
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self.predictor_norm = norm_layer(embed_dim)

        self.init_std = init_std
        #trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, ctx, enc_mask_indices, pred_mask_indices):
        """
        ctx: (B, v, E)
        enc_mask_indices: [(B, ei,), (B, ei)]
        pred_mask_indices: [(B, pi,), (B, pi)]
        """

        ei = torch.concat(enc_mask_indices, dim=0)   # (2B, e)
        pi = torch.concat(pred_mask_indices, dim=0) # (2B, p)

        # Map to predictor space
        #x = self.predictor_embed(ctx)
        #B, _, D = ctx.shape
        ctx_pooled = self.ctx_pool(ctx.mean(dim=1))
        B, E = ctx_pooled.shape
        ctx_expanded = ctx_pooled.unsqueeze(1).expand(B, pi.size(1), E)

        # Add positional embeddings to visible tokens
        pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        #ctx = ctx + apply_masks(pos_embed, ei)
        
        x = ctx_expanded + apply_masks(pos_embed, pi)


        # Create masked prediction tokens
        #pred_pos = apply_masks(pos_embed, pi)
        #pred_tokens = self.mask_token.repeat(
        #    pred_pos.size(0), pred_pos.size(1), 1
        #)
        #pred_tokens = pred_tokens + pred_pos

        # Concatenate context + masked tokens
        #x = torch.cat([ctx, pred_tokens], dim=1)
        #ctx_size = ctx.size(1)

        # ---- Token-wise JEPA prediction ----
        x = self.predictor_mlp(x)

        # Keep only predictions for masked tokens
        #x = x[:, ctx_size:]

        # Project to encoder embedding dimension
        z = self.predictor_proj(x)

        if self.predictor_norm is not None:
            z = self.predictor_norm(z)

        return z

