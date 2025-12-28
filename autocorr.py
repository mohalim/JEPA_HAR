import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from data.dataset import SequentialJEPASensorDataset
from models.transformer_encoder import PatchTransformerEncoder

def autocorr_conv1d_multi(x):
    # Handle different input shapes
    if x.ndim == 1:
        x = x.view(1, 1, -1)
    elif x.ndim == 2:
        x = x.unsqueeze(0)  # Add batch dimension
    
    B, T, C = x.shape # Batch, Patch Length, Channels

    x = x.view(B, C, T)
    
    # Time-reversed copy
    x_rev = torch.flip(x, dims=[2])
    
    # Reshape to merge batch and channel dimensions
    # Input: (1, B*C, T) - single batch with B*C channels
    # Weight: (B*C, 1, T) - B*C filters, each processing 1 channel
    x_flat = x.view(1, B * C, T)
    x_rev_flat = x_rev.view(B * C, 1, T)
    
    # Full correlation using grouped conv1d
    # groups=B*C: each of the B*C channels convolves with its own filter
    R_full = F.conv1d(x_flat, x_rev_flat, padding=T-1, groups=B*C)
    
    # Reshape back to (B, C, 2T-1) and keep non-negative lags only
    R_full = R_full.view(B, C, -1)
    R = R_full[:, :, T-1:]  # lags 0..N-1
    
    # Normalize by lag-0 value
    R = R / R[:, :, 0:1]
    
    # Find argmax for each channel (excluding lag 0)
    # peaks = torch.argmax(R[:, :, 1:], dim=2) + 1
    
    return R.view(B,T,C) #, peaks


def autocorr_conv1d_patch_multi(x):
    # Handle different input shapes
    if x.ndim == 1:
        x = x.view(1, 1, 1, -1)  # (B, N, T, C)
    elif x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # Add batch and patch dimensions
    elif x.ndim == 3:
        x = x.unsqueeze(1)  # Add patch dimension (assuming input is B, T, C)
    
    B, N, T, C = x.shape  # Batch, Num Patches, Patch Length, Channels
    
    # Reshape to merge batch and patch dimensions for parallel processing
    # Treat each (batch, patch) combination as an independent sample
    x_flat = x.view(B * N, T, C)
    
    # Transpose to (B*N, C, T) for conv1d
    x_flat = x_flat.transpose(1, 2).contiguous()  # (B*N, C, T)
    
    # Time-reversed copy
    x_rev = torch.flip(x_flat, dims=[2]).contiguous()
    
    # Reshape for grouped convolution
    # Input: (1, B*N*C, T) - single batch with B*N*C channels
    # Weight: (B*N*C, 1, T) - B*N*C filters, each processing 1 channel
    x_input = x_flat.view(1, B * N * C, T)
    x_weight = x_rev.view(B * N * C, 1, T)
    
    # Full correlation using grouped conv1d
    # groups=B*N*C: each of the B*N*C channels convolves with its own filter
    R_full = F.conv1d(x_input, x_weight, padding=T-1, groups=B*N*C)
    
    # Reshape back to (B*N, C, 2T-1) and keep non-negative lags only
    R_full = R_full.view(B * N, C, -1)
    R = R_full[:, :, T-1:]  # lags 0..T-1
    
    # Normalize by lag-0 value
    R = R / R[:, :, 0:1]
    
    # Reshape back to original batch and patch structure
    R = R.view(B, N, C, T)
    
    # Transpose to match input format (B, N, T, C)
    R = R.transpose(2, 3)  # (B, N, T, C)
    
    return R


def autocorr_similarity():
    pass

def projected_cosine_similarity(z1, z2, proj):
    """
    Projected cosine similarity (per-dimension).

    Args:
        z1, z2: tensors of shape (..., channels)
        proj: nn.Linear(channels, channels, bias=False)

    Returns:
        s: per-dimension cosine similarity contributions, shape (..., channels)
    """
    z1_p = F.normalize(proj(z1), dim=-1)
    z2_p = F.normalize(proj(z2), dim=-1)

    return z1_p * z2_p



num_patches = 5
window_size = 100
channels = 6
patch_dim = (window_size // num_patches) * channels
embedding_dim = 120
hidden_dim = 256

context_encoder = PatchTransformerEncoder(patch_dim, embedding_dim, num_patches)

dataset = SequentialJEPASensorDataset(
    root_dir="data/SBRHAPT/Train/",
    window_size=100,
    overlap=0.5,
    num_patches=5,
    mask_ratio=0.3
)

for data in dataset:
    '''
    seqs = data["all_patches"]
    N, E = seqs[0].shape
    seq1 = seqs[0]      # [N, E]
    seq2 = seqs[1]      # [N, E]

    patch_indices = torch.arange(N)
    z1 = context_encoder(seq1.unsqueeze(0), patch_indices) # [B, N, E]
    t = int(E / channels)   # 120/6 = 20
    z1 = z1.view(1, N, t, channels)
    z1 = z1.view(1, N*t, channels)

    a1 = autocorr_conv1d_multi(z1)

    patch_indices = torch.arange(N*2)
    seq12 = torch.cat((seq1, seq2), dim=0)
    z = target_encoder(seq12.unsqueeze(0), patch_indices) # [B, N*2, E]'''

    # Treat patches separately
    visible_patches = data["visible_patches"]
    visible_idx = data["visible_idx"]
    
    x = torch.cat((visible_patches[0], visible_patches[1]), dim=0) # concat two windows along patch dimension
    x_idx = torch.cat((visible_idx[0], visible_idx[1]))
    z = context_encoder(x.unsqueeze(0), x_idx) # (B, N, E) - Batch, Num of Patches, Embedding Size
    len_z1 = len(visible_patches[0])

    B, N, E = z.shape   # (B, N, E) - Batch, Num of Patches, Embedding Size
    t = int(E/channels)
    z = z.view(B, N, t, channels)
    z1 = z[:,:len_z1,:,:]
    z2 = z[:,len_z1:,:,:]

    proj = torch.nn.Linear(channels, channels, bias=False)
    s = projected_cosine_similarity(z2, z1, proj)

    a = autocorr_conv1d_patch_multi(z)
    a1 = a[:,:len_z1,:,:]
    a2 = a[:,len_z1:,:,:]
    fd = a2 - a1



    break
