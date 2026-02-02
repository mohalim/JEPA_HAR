# training/losses.py
import torch
import torch.nn.functional as F

def invariance_loss(z1, z2):
    return F.smooth_l1_loss(z1, z2)

def norm_loss(z, target=1.0):
    norm = z.norm(dim=-1)
    return ((norm - target) ** 2).mean()

def variance_loss(z, gamma=1.0, eps=1e-4):
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
    z1,
    z2,
    sim_coeff=1.0, # Sensor embeddings have lower semantic richness than vision
    norm_coeff=0.01,
    var_coeff=1.0,  
    cov_coeff=0.1,  # based on https://arxiv.org/abs/2410.19560
):
    #z1 = F.normalize(z1, dim=-1)
    #z2 = F.normalize(z2, dim=-1)

    sim = invariance_loss(z1, z2)
    norm_pred = norm_loss(z1)
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

def vicreg_jepa_loss(
    z1,     # z_pred
    z2,     # z_tgt
    z3,     # z_ctx
    sim_coeff=10.0, # Sensor embeddings have lower semantic richness than vision
    norm_coeff=1.0,
    var_coeff=10.0,  
    cov_coeff=1.0,  # based on https://arxiv.org/abs/2410.19560
    ctx_var_coeff=0.0,  
    ctx_cov_coeff=0.0
):
    #z1 = F.normalize(z1, dim=-1)
    #z2 = F.normalize(z2, dim=-1)

    sim = invariance_loss(z1, z2)
    var_pred = variance_loss(z1)
    norm_pred = norm_loss(z1)
    var_target = variance_loss(z2)
    cov_pred = covariance_loss(z1)
    cov_target = covariance_loss(z2)

    # VICReg regularization on context encoder outputs
    var_ctx = variance_loss(z3)
    cov_ctx = covariance_loss(z3)

    # Combined loss
    loss = (
        sim_coeff * sim +
        norm_coeff * norm_pred +
        var_coeff * (var_pred + var_target) +
        cov_coeff * (cov_pred + cov_target) +
        ctx_var_coeff * var_ctx +  
        ctx_cov_coeff * cov_ctx     
    )

    return loss, {
        'sim': sim.item(),
        'var_pred': var_pred.item(),
        'var_target': var_target.item(),
        'var_ctx': var_ctx.item(),  
        'cov_pred': cov_pred.item(),
        'cov_target': cov_target.item(),
        'cov_ctx': cov_ctx.item(),  
    }

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