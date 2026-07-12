# training/losses.py
import torch
import torch.nn.functional as F

def invariance_loss(z1, z2):
    return F.smooth_l1_loss(z1, z2)

def norm_loss(z, embed_dim, sigma=0.30):
    target = 1 + sigma * (embed_dim**0.5 - 1)
    norm = z.norm(dim=-1)
    return ((norm - target) ** 2).mean()

# Sensor data is inherently less "diverse" than other unstructured data e.g. high-res images
# So gamma is set to 0.4 - 0.5. 
# Setting gamma=1.0 might forcing the model to produce noise just to satisfy the loss
def variance_loss(z, gamma=0.5, eps=1e-4):
    z = z.reshape(-1, z.shape[-1])      # (B*seq_len, E)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)    # (B*seq_len, E)
    #std = torch.sqrt(z.var(dim=1, unbiased=False) + eps)  # per-sample
    return torch.mean(F.relu(gamma - std))

def covariance_loss(z):
    z = z.reshape(-1, z.shape[-1])  # (B * T, E)
    N, E = z.shape

    z = z - z.mean(dim=0)
    z = z / (z.std(dim=0) + 1e-4)

    cov = (z.T @ z) / (N - 1)          # (E, E)
    
    # Total sum of squared elements
    cov_total = (cov ** 2).sum()
    
    # Subtract diagonal contribution
    cov_diag = (torch.diagonal(cov) ** 2).sum()
    
    # Off-diagonal sum normalized by E
    loss = (cov_total - cov_diag) / E
    
    return loss

def vicreg_loss(
    z1, # predict
    z2, # target
    embed_dim,
    sim_coeff=1.0, # Sensor embeddings have lower semantic richness than vision
    norm_coeff=0.1,
    var_coeff=5.0,  # how much the model cares about maintaining variance
    cov_coeff=0.2,  # based on https://arxiv.org/abs/2410.19560 - it shouldn't be too strong that it outweighs the actual learning (invariance)
):
    #z1 = F.normalize(z1, dim=-1)
    #z2 = F.normalize(z2, dim=-1)

    sim = invariance_loss(z1, z2)
    norm_pred = norm_loss(z1, embed_dim)
    # norm_target = norm_loss(z2, embed_dim)
    var_pred = variance_loss(z1)
    var_target = variance_loss(z2)
    cov_pred = covariance_loss(z1)
    cov_target = covariance_loss(z2)

    loss = (
        sim_coeff * sim + 
        norm_coeff * norm_pred + 
        var_coeff * (var_pred + var_target) + 
        cov_coeff * (cov_pred + cov_target)
        )
    
    return loss, {
        'sim': sim.item(),
        'var_pred': var_pred.item(),
        'var_target': var_target.item(),
        'cov_pred': cov_pred.item(),
        'cov_target': cov_target.item(),
        }
