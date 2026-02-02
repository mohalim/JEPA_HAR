import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from torch.utils.data import DataLoader

from models.jepa_conv import JEPA_SEQ
from training.train import train_self_supervised
from utils.history import save_history_txt
from data.dataset_seq import SequentialSensorDataset

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

exp_num = "conv5_e384"
is_load = False
is_winseq = True

batch_size = 128

window_size = 102
overlap = 0.1
num_windows = 2
channels = 6
patch_size = 6             # window_size / patch_size = 17 patches
masking_factor = 0.5
reverse_ratio = 0.5
noise_std = 0.03

train_path_dataset = "../../Datasets/REALDISP_AccGyro/train_npy/"
val_path_dataset = "../../Datasets/REALDISP_AccGyro/val_npy/"

train_dataset = SequentialSensorDataset(
    root_dir=train_path_dataset,
    window_size=window_size,
    overlap=overlap,
    patch_size=patch_size,
    num_windows=num_windows,
    masking_factor=masking_factor,
    reverse_ratio=reverse_ratio,
    noise_std=noise_std
)

val_dataset = SequentialSensorDataset(
    root_dir=val_path_dataset,
    window_size=window_size,
    overlap=overlap,
    patch_size=patch_size,
    num_windows=num_windows,
    masking_factor=masking_factor,
    reverse_ratio=reverse_ratio,
    noise_std=noise_std
)

train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=4,
                          persistent_workers=True,
                          pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# embedding dimension of convolutional encoder
kernel_sizes = [7, 5, 3, 3]
embedding_dim = 384
predictor_embed_dim = 192
predictor_n_heads = 4
predictor_n_layers = 2

model = JEPA_SEQ(
    seq_length=window_size,
    channels=channels,
    patch_size=patch_size,
    conv_kernel_sizes=kernel_sizes,
    #num_windows=num_windows, 
    embedding_dim=embedding_dim,
    predictor_embed_dim=predictor_embed_dim,
    predictor_n_heads=predictor_n_heads,
    predictor_n_layers=predictor_n_layers,
    is_seq=is_winseq
    ).to(device)

if not is_winseq:
    checkpoint_dir = f"checkpoints/w{window_size}/no_seq"
else: 
    checkpoint_dir = f"checkpoints/w{window_size}/win_seq_{exp_num}"

max_epochs = 100
patience = 20
base_lr = 1e-4
max_lr = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.99))

if is_load:
    start_epoch = 100
    checkpoint_file = 'checkpoint_epoch_125_w80_ed256_hd384.pt'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    history = train_self_supervised(model, train_loader, val_loader, optimizer,
                                    device, start_epoch=start_epoch, max_epochs=max_epochs,
                                    base_lr=base_lr, max_lr=max_lr, 
                                    checkpoint_dir=checkpoint_dir, patience=patience)
else:
    history = train_self_supervised(model, train_loader, val_loader, optimizer, 
                                    device, start_epoch=1, max_epochs=max_epochs, 
                                    base_lr=base_lr, max_lr=max_lr,
                                    checkpoint_dir=checkpoint_dir, patience=patience)

if not is_winseq:
    history_file_path = 'history/no_seq'
else:
    history_file_path = 'history/win_seq'
    
os.makedirs(history_file_path, exist_ok=True)
history_fname = f"history_w{window_size}_ed{embedding_dim}.txt"
save_history_txt(history, os.path.join(history_file_path, history_fname))

# Best checkpoint: 