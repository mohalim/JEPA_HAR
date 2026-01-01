import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from torch.utils.data import DataLoader

from models.jepa import JEPA_SEQ
from training.train import train_self_supervised
from utils.history import save_history_txt
from data.dataset_seq import SequentialSensorDataset

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

is_load = False

# 32:	3e-5
# 64:	6e-5 – 1e-4
# 128:	1e-4 – 2e-4
# 256:	2e-4 – 3e-4
lr = 5e-5
momentum = 0.95
batch_size = 128

num_windows = 2
window_size = 102
channels = 6
patch_size = 17     # window_size / patch_size = 6 patches
masking_factor = 0.5

train_path_dataset = "../../Datasets/REALDISP_AccGyro/train_npy/"
val_path_dataset = "../../Datasets/REALDISP_AccGyro/val_npy/"
test_path_dataset = "../../Datasets/REALDISP_AccGyro/test_npy/"

train_dataset = SequentialSensorDataset(
    root_dir=train_path_dataset,
    window_size=window_size,
    overlap=0.5,
    patch_size=patch_size,
    num_windows=num_windows,
    masking_factor=masking_factor,
    has_label=False
)

val_dataset = SequentialSensorDataset(
    root_dir=val_path_dataset,
    window_size=window_size,
    overlap=0.5,
    patch_size=patch_size,
    num_windows=num_windows,
    masking_factor=masking_factor,
    has_label=False
)

test_dataset = SequentialSensorDataset(
    root_dir=test_path_dataset,
    window_size=window_size,
    overlap=0.5,
    patch_size=patch_size,
    num_windows=num_windows,
    masking_factor=masking_factor,
    has_label=False
)

train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=4,
                          persistent_workers=True,
                          pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# embedding dimension of transformer encoder
embedding_dim = 256
predictor_embed_dim = 128
model = JEPA_SEQ(
    seq_length=window_size,
    channels=channels,
    patch_size=patch_size,
    num_windows=num_windows, 
    embedding_dim=embedding_dim, 
    predictor_embed_dim=predictor_embed_dim
    ).to(device)

max_epochs = 100
patience = 100
checkpoint_dir = f"checkpoints/w{window_size}"
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
if is_load:
    start_epoch = 100
    max_epochs = 100
    checkpoint_file = 'checkpoint_epoch_125_w80_ed256_hd384.pt'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    history = train_self_supervised(model, train_loader, val_loader, optimizer, scheduler,
                                    device, start_epoch=start_epoch, max_epochs=max_epochs, 
                                    checkpoint_dir=checkpoint_dir, patience=patience)
else:
    history = train_self_supervised(model, train_loader, val_loader, optimizer, 
                                    scheduler, device, max_epochs=max_epochs, 
                                    checkpoint_dir=checkpoint_dir, patience=patience)

history_file_path = 'history'
os.makedirs(history_file_path, exist_ok=True)
history_fname = f"history_w{window_size}_ed{embedding_dim}.txt"
save_history_txt(history, os.path.join(history_file_path, history_fname))

# Best checkpoint: 