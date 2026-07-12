import torch.nn as nn
import torch

class JEPAClassifier(nn.Module):
    def __init__(self, context_encoder, classifier, pooling="max", freeze_encoder=True):
        super().__init__()

        pools = {
            "mean": lambda x: x.mean(dim=1),
            "max":  lambda x: x.max(dim=1).values,
        }
        assert pooling in pools
        self.pool = pools[pooling]

        self.context_encoder = context_encoder  # use pretrained encoder
        # simple classifier head
        self.classifier = classifier

        if freeze_encoder:
            self.freeze_encoder()

    def get_layer_groups(self):
        if hasattr(self.context_encoder, "blocks"):
            # Assuming blocks is a nn.ModuleList or list of layers
            return list(self.context_encoder.blocks)
        else:
            raise AttributeError("Encoder must have a 'blocks' attribute containing sequential layers.")

    def update_optimizer_groups(self, optimizer, base_clf_lr, decay_factor=0.1):
        blocks = self.get_layer_groups()
        num_blocks = len(blocks)

        # Optimizer configuration states (fused, amsgrad, etc.)
        opt_defaults = {k: v for k, v in optimizer.defaults.items()}
        
        # 1. Base Classification Head Parameters
        clf_group = {
            **opt_defaults,
            "params": [p for p in self.classifier.parameters() if p.requires_grad],
            "lr": base_clf_lr,
            "betas": (0.9, 0.99)
        }
        new_param_groups = [clf_group]

        # 2. Final Normalization Layer (Top level target)
        if hasattr(self.context_encoder, "norm") and self.context_encoder.norm is not None:
            norm_params = [p for p in self.context_encoder.norm.parameters() if p.requires_grad]
            if norm_params:
                new_param_groups.append({
                    **opt_defaults,
                    "params": norm_params,
                    "lr": base_clf_lr * decay_factor,
                    "betas": (0.9, 0.99)
                })

        # 3. Convolutional Blocks (Layer-wise decaying learning rates)
        for idx, block in enumerate(blocks):
            active_params = [p for p in block.parameters() if p.requires_grad]
            if active_params:
                layer_depth = num_blocks - idx
                layer_lr = base_clf_lr * (decay_factor ** layer_depth)
                
                new_param_groups.append({
                    **opt_defaults,
                    "params": active_params,
                    "lr": layer_lr,
                    "betas": (0.9, 0.99)
                })

        # Swap out optimizer states without dropping historic state momentum
        optimizer.param_groups = new_param_groups

    def freeze_encoder(self):
        for p in self.context_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.context_encoder.parameters():
            p.requires_grad = True

    def unfreeze_last_k_layers(self, k):
        for p in self.context_encoder.parameters():
            p.requires_grad = False

        # 2. Unfreeze last k ConvBlocks
        if hasattr(self.context_encoder, "blocks"):
            for block in self.context_encoder.blocks[-k:]:
                for p in block.parameters():
                    p.requires_grad = True
        else:
            raise AttributeError("Encoder has no attribute 'blocks'")

        # Unfreeze final norm
        if hasattr(self.context_encoder, "norm") and self.context_encoder.norm is not None:
            for p in self.context_encoder.norm.parameters():
                p.requires_grad = True

    def forward(self, x):
        # full_mask = torch.ones_like(x1[:,:,0], dtype=torch.bool)
        x_tokens = self.context_encoder(x) # (2B, N, E)

        # split the Batch dimension (0)
        x1, x2 = torch.chunk(x_tokens, chunks=2, dim=0)   # (B, N, E) and (B, N, E)
        x_tokens = torch.cat([x1, x2], dim=1)             # (B, 2N, E)

        # x_tokens = x_tokens.mean(dim=1)   # (B, E)
        x_tokens = self.pool(x_tokens)

        # Forward through classifier head
        return self.classifier(x_tokens)         # [B, num_classes]

