import torch.nn as nn
import torch

class JEPALinearProbe(nn.Module):
    """
    Linear probing classifier on top of a pretrained JEPA context encoder
    for sensor-based HAR.
    """

    def __init__(
        self,
        context_encoder,
        embed_dim,
        num_classes,
        freeze_encoder=True,
        pooling="mean",  # "mean" or "cls" (if supported)
    ):
        super().__init__()

        self.context_encoder = context_encoder
        self.pooling = pooling

        # Linear probing head (no MLP, no non-linearity)
        self.classifier = nn.Linear(embed_dim, num_classes)

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self):
        """Freeze all encoder parameters (standard linear probing)."""
        for p in self.context_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        """Optional: for fine-tuning after linear probing."""
        for p in self.context_encoder.parameters():
            p.requires_grad = True

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, C)
               Single-view sensor sequence for HAR
        Returns:
            logits: (B, num_classes)
        """

        # Encoder output: (2B, N, E)
        x_tokens = self.context_encoder(x)

        # split the Batch dimension (0)
        x1, x2 = torch.chunk(x_tokens, chunks=2, dim=0)   # (B, N, E) and (B, N, E)
        x_tokens = torch.cat([x1, x2], dim=1)             # (B, 2N, E)

        # Pooling
        if self.pooling == "mean":
            features = x_tokens.mean(dim=1)      # (B, E)
        elif self.pooling == "cls":
            features = x_tokens[:, 0]            # (B, E)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Linear classifier
        logits = self.classifier(features)
        return logits


