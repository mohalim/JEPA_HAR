import torch.nn as nn
import torch

class JEPAClassifier(nn.Module):
    def __init__(self, context_encoder, classifier, freeze_encoder=True):
        super().__init__()

        self.context_encoder = context_encoder  # use pretrained encoder
        # simple classifier head
        self.classifier = classifier

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self):
        for p in self.context_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.context_encoder.parameters():
            p.requires_grad = True

    def unfreeze_last_k_layers(self, k):
        """
        Unfreeze last k TransformerEncoder layers.
        """
        for p in self.context_encoder.parameters():
            p.requires_grad = False

        # 2. Unfreeze last k ConvBlocks
        if hasattr(self.context_encoder, "blocks"):
            for block in self.context_encoder.blocks[-k:]:
                for p in block.parameters():
                    p.requires_grad = True
        else:
            raise AttributeError("Encoder has no attribute 'blocks'")

        # Always unfreeze final norm
        if hasattr(self.context_encoder, "norm") and self.context_encoder.norm is not None:
            for p in self.context_encoder.norm.parameters():
                p.requires_grad = True

    def forward(self, x):
        """
        batch keys:
          x: [(B, seq_len, C), (B, seq_len, C)]
        """
        # full_mask = torch.ones_like(x1[:,:,0], dtype=torch.bool)
        x_tokens = self.context_encoder(x) # (2B, N, E)

        # split the Batch dimension (0)
        x1, x2 = torch.chunk(x_tokens, chunks=2, dim=0)   # (B, N, E) and (B, N, E)
        x_tokens = torch.cat([x1, x2], dim=1)             # (B, 2N, E)

        x_tokens = x_tokens.mean(dim=1)   # (B, E)

        # Forward through classifier head
        return self.classifier(x_tokens)         # [B, num_classes]

