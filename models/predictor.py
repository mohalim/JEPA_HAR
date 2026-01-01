# models/predictor.py
import torch
import torch.nn as nn

from utils.misc import trunc_normal_, get_1d_sincos_pos_embed, apply_masks

class Predictor(nn.Module):
    def __init__(
        self,
        num_patches=17,
        embed_dim=256,
        predictor_embed_dim=128,
        n_heads=2,
        n_layers=1,
        dropout=0.1,
        init_std=0.02,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # Predictor positional embedding (1D)
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False)
        predictor_pos_embed = get_1d_sincos_pos_embed(
            self.predictor_pos_embed.shape[-1],
            num_patches,
            cls_token=False
        )
        self.predictor_pos_embed.data.copy_(torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))

        layer = nn.TransformerEncoderLayer(
            d_model=predictor_embed_dim,
            nhead=n_heads,
            dim_feedforward=predictor_embed_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
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

    def forward(self, x, enc_mask_indices, pred_mask_indices):
        """
        x: (2B, v, E) - context embeddings for window 1 and window 2, the embeddings are stacked along Batch dimension
        enc_mask_indices: (B, ei,) - batch, visible indices
        pred_mask_indices: (B, pi,) - batch, predict indices
        """
        assert (enc_mask_indices is not None) and (pred_mask_indices is not None), \
            'Cannot run predictor without mask indices'
        
        ei = torch.concat(enc_mask_indices, dim=0)      # (2B, e)
        pi = torch.concat(pred_mask_indices, dim=0)     # (2B, p)
        
        # Batch Size
        # B = int(B / 2)

        # Map from encoder-dim to predictor-dim
        x = self.predictor_embed(x)
        B, _, E = x.shape
        
        # Add positional embedding to context (x) embeddings
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)      # (B, N, E)
        x += apply_masks(x_pos_embed, ei)                           # (B, e, E)

        # Add positional embedding to masked (target) embeddings
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)         # (B, N, E)
        pos_embs = apply_masks(pos_embs, pi)                        # (B, p, E)

        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1) # (B, p, E)
        pred_tokens += pos_embs

        _, ctx_size, _ = x.shape

        # Concat mask tokens with x
        x = torch.cat([x, pred_tokens], dim=1)

        x = self.transformer(x)

        x = self.predictor_norm(x)

        # Return preds for mask tokens
        x = x[:, ctx_size:]
        z = self.predictor_proj(x)

        return z

