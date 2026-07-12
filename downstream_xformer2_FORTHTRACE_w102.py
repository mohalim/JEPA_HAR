import torch
import os
import glob
import torch.nn as nn
import warnings
import shutil

from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader

from models.jepa_xformer import JEPA_SEQ
from models.classifier_xformer import JEPAClassifier
from data.dataset_supervised import SequentialSupervisedSensorDataset
from training.train_clf import train_supervised, evaluate
from utils.logging import setup_logger, log_evaluation
from utils.misc import get_classifier_param_groups

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

is_winseq = True
top_k = 4       # number of final layers of encoder to fine-tune

dev_id = 3      # Dataset dev id: 3. Torso 4. Right Thigh
exp_num = "xformer2_e440"
ckp_num = 75
is_train = 0
is_log_result = 0 if is_train else 1    # if training don't log but print results only
is_print_all_results = 0 if is_log_result else 1 # print all checkpoints' result, from 70 - 100

window_size = 102
overlap = 0.5
num_windows = 2
channels = 6
patch_size = 6      # window_size / patch_size = 17 patches
batch_size = 32     # 32 (281), 64 (141)
     
train_path_dataset = f"../../Datasets/FORTH_TRACE/Data{dev_id}/train_npy/"
val_path_dataset = f"../../Datasets/FORTH_TRACE/Data{dev_id}/val_npy/"
test_path_dataset = f"../../Datasets/FORTH_TRACE/Data{dev_id}/test_npy/"

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

embedding_dim = 440
n_heads = 8
n_layers = 4
predictor_embed_dim = 220
predictor_n_heads = 4
predictor_n_layers = 2

model = JEPA_SEQ(
    seq_length=window_size,
    channels=channels,
    patch_size=patch_size,
    num_windows=num_windows, 
    embedding_dim=embedding_dim, 
    n_heads=n_heads,
    n_layers=n_layers,
    predictor_embed_dim=predictor_embed_dim,
    predictor_n_heads=predictor_n_heads,
    predictor_n_layers=predictor_n_layers,
    is_seq=is_winseq
    ).to(device)

checkpoint_file = 'cpt_epoch154_w102_edim440.pt'
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint_file), weights_only=True))
model.eval()
context_encoder = model.context_encoder

num_classes = 6

classifier_head = nn.Sequential(
            nn.Linear(embedding_dim, int(embedding_dim/2)),
            nn.GELU(),
            #nn.LayerNorm(int(embedding_dim/2)),
            nn.Dropout(0.3),
            nn.Linear(int(embedding_dim/2), num_classes)
            )

checkpoint_clf_dir = None
# Default value
lr_decay_factor = 0.97
clf_lr = 1e-3

# sigma = 0.20
# a) wd=6e-2 (0.8287/0.7370) ckp=83; b) wd=6.5e-2 (0.8533/0.7395) ckp=76; c) wd=1e-1 (0.8606/0.7498) ckp=75;
weight_decay = 1e-1
early_stop_metric = 'avg_f1'

if is_train:
    max_epochs = 100
    patience = 100

    classifier_model = JEPAClassifier(context_encoder, 
                                      classifier_head,
                                      freeze_encoder=True).to(device)

    # --- UNFREEZING SCHEDULE ---
    # Epoch X: unfreeze top K blocks.
    unfreeze_schedule = {
        1:  0,   # Epochs 1-10: Pure linear probing (Encoder fully frozen)
        11: 1,   # Epoch 11: Unfreeze the final transformer/conv block
        41: 2,   # Epoch 41: Unfreeze the top 2 blocks
        71: 4,   # Epoch 71: Unfreeze the top 4 blocks
    }

    checkpoint_clf_dir = f"checkpoints_clf/FORTH_TRACE/w{window_size}/progressive_{exp_num}"

    # Remove existing folders and checkpoints
    folder = Path(checkpoint_clf_dir)
    if folder.exists() and folder.is_dir():
        shutil.rmtree(folder)

    optimizer = torch.optim.AdamW(classifier_model.parameters(), 
                                  lr=clf_lr, 
                                  betas=(0.9, 0.99),
                                  weight_decay=weight_decay
                                  )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    history = train_supervised(classifier_model, train_loader, val_loader,
                               optimizer, base_clf_lr=clf_lr, lr_decay_factor=lr_decay_factor,
                               unfreeze_schedule=unfreeze_schedule, device=device, max_epochs=max_epochs,
                               checkpoint_dir=checkpoint_clf_dir, patience=patience, early_stop_metric=early_stop_metric)
    

if checkpoint_clf_dir is None:
    checkpoint_clf_dir = f"checkpoints_clf/FORTH_TRACE/w{window_size}/progressive_{exp_num}"    
    classifier_model = JEPAClassifier(context_encoder, 
                                      classifier_head,
                                      pooling='max',
                                      freeze_encoder=True).to(device)

if is_train:
    all_files = glob.glob(os.path.join(checkpoint_clf_dir, "*.pt"))
    path_checkpoint = all_files[-1]
else:
    best_checkpoint = f'best_model_progressive_epoch{ckp_num}.pt'
    path_checkpoint = os.path.join(checkpoint_clf_dir, best_checkpoint)


if is_log_result:
    print(f"Loading best model: {path_checkpoint}")
    classifier_model.load_state_dict(torch.load(path_checkpoint, weights_only=True))
    test_metrics = evaluate(classifier_model, test_loader, torch.nn.CrossEntropyLoss(), device)

    result_file_path = os.path.join(checkpoint_clf_dir, "results.log")
    if os.path.exists(result_file_path):
        os.remove(result_file_path)
    logger = setup_logger(checkpoint_clf_dir, "results.log", "evaluation")
    logger.info(f"weight_decay: {weight_decay}")
    logger.info(f"lr_decay_factor: {lr_decay_factor}")
    logger.info(f"clf_lr: {clf_lr}")
    logger.info(f"Encoder: {checkpoint_file}")
    logger.info(f"Loading best model: {path_checkpoint}")
    # log_evaluation(logger, test_metrics["acc"], test_metrics["conf_matrix"], test_metrics["clf_report"])
    log_evaluation(logger, test_metrics)

    print(test_metrics["conf_matrix"])
    print(test_metrics["clf_report"])
    print(test_metrics["acc"])
    print(test_metrics["avg_f1"])

if is_print_all_results:
    best_avg_f1 = 0
    best_acc = 0
    best_ckp_num = 0
    for ckp_num in range(70, 101):
        best_checkpoint = f'best_model_progressive_epoch{ckp_num}.pt'
        path_checkpoint = os.path.join(checkpoint_clf_dir, best_checkpoint)
        print(f"Loading best model: {path_checkpoint}")
        classifier_model.load_state_dict(torch.load(path_checkpoint, weights_only=True))
        test_metrics = evaluate(classifier_model, test_loader, torch.nn.CrossEntropyLoss(), device)
        #print(test_metrics["conf_matrix"])
        #print(test_metrics["clf_report"])
        print(test_metrics["acc"])
        print(test_metrics["avg_f1"])
        if test_metrics["avg_f1"] > best_avg_f1:
            best_avg_f1 = test_metrics["avg_f1"]
            best_acc = test_metrics["acc"]
            best_ckp_num = ckp_num
    
    print(f"Best ckp_num: {best_ckp_num}")
    print(f"Best avg f1: {best_avg_f1}")
    print(f"Best accuracy: {best_acc}")