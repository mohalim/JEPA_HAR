import torch.nn as nn
import torch

class JEPALinearProbe(nn.Module):
    def __init__(
        self,
        context_encoder,
        embed_dim,
        num_classes,
        freeze_encoder=True,
        pooling="max",  # "mean" or "cls" (if supported)
    ):
        super().__init__()

        pools = {
            "mean": lambda x: x.mean(dim=1),
            "max":  lambda x: x.max(dim=1).values,
        }

        self.context_encoder = context_encoder
        self.pool = pools[pooling]

        # Linear probing head (no MLP, no non-linearity)
        self.classifier = nn.Linear(embed_dim, num_classes)

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self):
        """ Freeze all encoder parameters (standard linear probing). """
        for p in self.context_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        """ Optional: for fine-tuning after linear probing. """
        for p in self.context_encoder.parameters():
            p.requires_grad = True

    def forward(self, x):
        """
        x: Tensor of shape (B, T, C)
        """
        # Encoder output: (2B, N, E)
        x_tokens = self.context_encoder(x)

        # split the Batch dimension (0)
        x1, x2 = torch.chunk(x_tokens, chunks=2, dim=0)   # (B, N, E) and (B, N, E)
        x_tokens = torch.cat([x1, x2], dim=1)             # (B, 2N, E)
    
        features = self.pool(x_tokens)

        # Linear classifier
        logits = self.classifier(features)
        return logits


