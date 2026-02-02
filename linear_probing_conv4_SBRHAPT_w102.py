import torch
import os
import glob
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings

from models.jepa_conv import JEPA_SEQ
from models.linear_probing import JEPALinearProbe
from data.dataset_supervised import SequentialSupervisedSensorDataset
from training.train_clf import train_supervised, evaluate
from utils.logging import setup_logger, log_evaluation

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

is_winseq = True

exp_num = "conv4_e256"

window_size = 102
overlap = 0.5
num_windows = 2
channels = 6
patch_size = 6      # window_size / patch_size = 17 patches
batch_size = 64
     

train_path_dataset = "../../Datasets/SBRHAPT/train_npy/"
val_path_dataset = "../../Datasets/SBRHAPT/val_npy/"
test_path_dataset = "../../Datasets/SBRHAPT/test_npy/"

train_dataset = SequentialSupervisedSensorDataset(
    root_dir=train_path_dataset,
    window_size=window_size,
    overlap=overlap,
    num_windows=num_windows
)

val_dataset = SequentialSupervisedSensorDataset(
    root_dir=val_path_dataset,
    window_size=window_size,
    overlap=overlap,
    num_windows=num_windows
)

test_dataset = SequentialSupervisedSensorDataset(
    root_dir=test_path_dataset,
    window_size=window_size,
    overlap=overlap,
    num_windows=num_windows
)

train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=4,
                          persistent_workers=True,
                          pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if not is_winseq:
    checkpoint_dir = f"checkpoints/w{window_size}/no_seq"
else:
    checkpoint_dir = f"checkpoints/w{window_size}/win_seq_{exp_num}"

kernel_sizes = [7, 5, 3, 3]
embedding_dim = 256
predictor_embed_dim = 128
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

checkpoint_file = 'cpt_epoch80_w102_edim256.pt'
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint_file), weights_only=True))
model.eval()
context_encoder = model.context_encoder

num_classes = 12
checkpoint_clf_dir = None

max_epochs = 100
patience = 100

linear_model = JEPALinearProbe(context_encoder, 
                               embed_dim=embedding_dim,
                               num_classes=num_classes,
                               freeze_encoder=True).to(device)

checkpoint_clf_dir = f"checkpoints_clf/w{window_size}/probe_{exp_num}"
clf_lr = 1e-3

optimizer = torch.optim.AdamW(linear_model.parameters(), lr=clf_lr, betas=(0.9, 0.99))

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
history = train_supervised(linear_model, train_loader, val_loader,
                           optimizer, scheduler, device, max_epochs=max_epochs,
                           checkpoint_dir=checkpoint_clf_dir, patience=patience, stage_two=False)

logger = setup_logger(checkpoint_clf_dir)

all_files = glob.glob(os.path.join(checkpoint_clf_dir, "*.pt"))
best_checkpoint = all_files[-1]
print(f"Loading best model: {best_checkpoint}")
linear_model.load_state_dict(torch.load(best_checkpoint, weights_only=True))

test_metrics = evaluate(linear_model, test_loader, torch.nn.CrossEntropyLoss(), device)
log_evaluation(logger, test_metrics["acc"], test_metrics["conf_matrix"], test_metrics["clf_report"])
print(test_metrics["acc"])
print(test_metrics["conf_matrix"])
print(test_metrics["clf_report"])

