import torch.nn as nn
import torch

class JEPAClassifier(nn.Module):
    def __init__(self, context_encoder, classifier, embedding_dim=128, num_classes=12, freeze_encoder=True):
        super().__init__()

        self.context_encoder = context_encoder  # use pretrained encoder

        if freeze_encoder:
            for param in self.context_encoder.parameters():
                param.requires_grad = False

        # simple classifier head
        self.classifier = classifier

    def forward(self, x1, x2):
        """
        batch keys:
          w1: (B, seq_len, C)
          w2:  (B, seq_len, C)
          label: (B, seq_len)
        """

        B, seq_len, C = x1.shape
        full_mask = torch.ones_like(x1[:,:,0], dtype=torch.bool)
        z_tokens = self.context_encoder(x1, x2, mask=full_mask) # (B, seq_len+seq_len, E)

        z_mean = z_tokens.mean(dim=1)   # (B, E)

        # Forward through classifier head
        return self.classifier(z_mean)         # [B, num_classes]

