# training/losses.py
import torch
import torch.nn.functional as F

def invariance_loss(z1, z2):
    z1 = F.normalize(z1, dim=-1)     # (B, seq_len, E)
    z2 = F.normalize(z2, dim=-1)     # (B, seq_len, E)
    return F.mse_loss(z1, z2)

def variance_loss(z, gamma=1.0, eps=1e-4):
    z = z.reshape(-1, z.shape[-1])      # (B, seq_len, E)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)    # (B*seq_len, E)
    return torch.mean(F.relu(gamma - std))

def covariance_loss(z):
    z = z.reshape(-1, z.shape[-1])  # (B*seq_len, E)
    B, E = z.shape

    z = z - z.mean(dim=0)
    z = z / (z.std(dim=0) + 1e-4)

    cov = (z.T @ z) / (B - 1)          # (E, E)
    
    # Total sum of squared elements
    cov_total = (cov ** 2).sum()
    
    # Subtract diagonal contribution
    cov_diag = (torch.diagonal(cov) ** 2).sum()
    
    # Off-diagonal sum normalized by E
    loss = (cov_total - cov_diag) / E
    
    return loss

def vicreg_loss(
    z1,
    z2,
    sim_coeff=1.0, # Sensor embeddings have lower semantic richness than vision
    var_coeff=25.0,  
    # cov_coeff=25.0,  # Need stronger decorrelation pressure because small cov_coeff lets embeddings collapse directionally
):
    sim = invariance_loss(z1, z2)
    var = variance_loss(z1) + variance_loss(z2)
    # cov = covariance_loss(z1) + covariance_loss(z2)

    loss = sim_coeff * sim + var_coeff * var # + cov_coeff * cov
    return loss, sim, var #, cov

'''
def cosine_sim_loss(z_pred, z_target):
    z_target = z_target.detach()

    # Use normalized embeddings for similarity
    z_pred_n = F.normalize(z_pred, dim=-1)
    z_target_n = F.normalize(z_target, dim=-1)

    return 1 - (z_pred_n * z_target_n).sum(dim=-1).mean()

def jepa_loss(sim_loss, var_loss, alpha=1.0, beta=25):
    return alpha * sim_loss + beta * var_loss
'''