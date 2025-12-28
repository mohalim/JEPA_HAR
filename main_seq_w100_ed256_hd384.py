import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from torch.utils.data import DataLoader

from models.jepa_pos import JEPA_SEQ
from training.train import train_self_supervised
from utils.history import save_history_txt
from data.dataset_seq import SequentialMaskingSensorDataset

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# 32:	3e-5
# 64:	6e-5 – 1e-4
# 128:	1e-4 – 2e-4
# 256:	2e-4 – 3e-4
lr = 5e-5
momentum = 0.95
batch_size = 128

num_windows = 2
window_size = 100
channels = 6

# output dimension of transformer encoder
embedding_dim = 256
hidden_dim = 384

# -----------------------------
# 1. JEPA self-supervised pretraining
# -----------------------------
model = JEPA_SEQ(
    input_channels=channels, embedding_dim=embedding_dim, 
    window_size=window_size, hidden_dim=hidden_dim
    ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_path_dataset = "../../Datasets/REALDISP_AccGyro/train_npy/"
val_path_dataset = "../../Datasets/REALDISP_AccGyro/val_npy/"
test_path_dataset = "../../Datasets/REALDISP_AccGyro/test_npy/"

train_dataset = SequentialMaskingSensorDataset(
    root_dir=train_path_dataset,
    window_size=100,
    overlap=0.5,
    masking_factor=0.5,
    has_label=False
)

val_dataset = SequentialMaskingSensorDataset(
    root_dir=val_path_dataset,
    window_size=100,
    overlap=0.5,
    masking_factor=0.5,
    has_label=False
)

test_dataset = SequentialMaskingSensorDataset(
    root_dir=test_path_dataset,
    window_size=100,
    overlap=0.5,
    masking_factor=0.5,
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

max_epochs = 100
patience = 100
history = train_self_supervised(model, train_loader, val_loader, optimizer, 
                                device, max_epochs=max_epochs, patience=patience)

history_file_path = 'history'
os.makedirs(history_file_path, exist_ok=True)
history_fname = f"history_w{window_size}_ed{embedding_dim}_hd{hidden_dim}.txt"
save_history_txt(history, os.path.join(history_file_path, history_fname))

# Best checkpoint: checkpoint_epoch_100_w100_ed256_hd384.pt