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
        predictor_n_heads=2,
        predictor_n_layers=1,
        dropout=0.1,
        init_std=0.02,
        norm_layer=nn.LayerNorm,
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

        # for modeling window sequence
        #self.window_embed = nn.Embedding(2, predictor_embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=predictor_embed_dim,
            nhead=predictor_n_heads,
            dim_feedforward=predictor_embed_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, predictor_n_layers)

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

    def forward(self, ctx, enc_mask_indices, pred_mask_indices):
        """
        ctx: (B, v, E) - context embeddings for both windows
        enc_mask_indices: [(B, ei,), (B, ei)] - batch, visible indices
        pred_mask_indices: (B, pi,), (B, pi)] - batch, predict indices
        """
        #assert (enc_mask_indices is not None) and (pred_mask_indices is not None), \
        #    'Cannot run predictor without mask indices'
        
        ei = torch.concat(enc_mask_indices, dim=0)      # (2B, e)
        pi = torch.concat(pred_mask_indices, dim=0)     # (2B, p)
        # ei = enc_mask_indices[0]
        # pi = pred_mask_indices[1]
        
        # Batch Size
        # B = int(B / 2)

        # Map from encoder-dim to predictor-dim
        ctx = self.predictor_embed(ctx)     # embeddings for both windows - stacked along B dimension
        B, _, E = ctx.shape             
        
        # Create and add positional embedding to context (x) embeddings
        ctx_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)      # (2B, N, E)
        ctx += apply_masks(ctx_pos_embed, ei)                         # (2B, e, E)
        # ctx += ctx_pos_embed

        # Add positional embedding to masked (target) embeddings
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)         # (2B, N, E)
        pos_embs = apply_masks(pos_embs, pi)                        # (2B, p, E)

        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1) # (B, p, E)
        pred_tokens += pos_embs

        # Concat mask tokens with x
        # concat along batch dim (if dim=0) possible if ei shape = pi shape, (4B, e/p, E)
        # concat along patch dimension (if dim=1) if ei shape != pi shape, (2B, e + p, E)
        x = torch.cat([ctx, pred_tokens], dim=1)        # (2B, e + p, E)
        
        B_ctx_mask, ctx_size, _ = ctx.shape

        '''
        # Add window embedding
        window_idx = torch.tensor([0, 1], device=x.device)
        win_emb = self.window_embed(window_idx)   # (2, E)

        win_emb = win_emb.repeat_interleave(B_ctx_mask, dim=0) # (2B, E)
        x = x + win_emb.unsqueeze(1)                # (4B, e/p, E)
        '''
        x = self.transformer(x)

        x = self.predictor_norm(x)

        # Return preds for mask tokens
        # if concat dim=0
        # x = x[B:]
        # if concat dim=1
        x = x[:, ctx_size:]
        
        z = self.predictor_proj(x)

        return z

