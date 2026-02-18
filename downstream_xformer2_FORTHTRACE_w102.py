import torch
import os
import glob
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings

from models.jepa_xformer import JEPA_SEQ
from models.classifier_xformer import JEPAClassifier
from data.dataset_supervised import SequentialSupervisedSensorDataset
from training.train_clf import train_supervised, evaluate
from utils.logging import setup_logger, log_evaluation

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

is_winseq = True
top_k = 4       # number of final layers of encoder to fine-tune

dev_id = 3      # Dataset dev id: 3. Torso 4. Right Thigh
exp_num = "xformer2_e440"
is_train = True
is_stage_two = True

window_size = 102
overlap = 0.5
num_windows = 2
channels = 6
patch_size = 6      # window_size / patch_size = 17 patches
batch_size = 64
     
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

checkpoint_file = 'cpt_epoch80_w102_edim440.pt'
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint_file), weights_only=True))
model.eval()
context_encoder = model.context_encoder

num_classes = 12
classifier_head = nn.Sequential(
            nn.Linear(embedding_dim, int(embedding_dim/2)),
            nn.GELU(),
            #nn.LayerNorm(int(embedding_dim/2)),
            nn.Dropout(0.1),
            nn.Linear(int(embedding_dim/2), num_classes)
            )

checkpoint_clf_dir = None

if is_train:
    max_epochs = 100
    patience = 100

    classifier_model = JEPAClassifier(context_encoder, 
                                      classifier_head,
                                      freeze_encoder=True).to(device)

    if not is_stage_two:
        checkpoint_clf_dir = f"checkpoints_clf/FORTH_TRACE/w{window_size}/stage1_{exp_num}"
        clf_lr = 1e-3
        
        optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=clf_lr, betas=(0.9, 0.99))

    else:
        checkpoint_clf_dir = f"checkpoints_clf/FORTH_TRACE/w{window_size}/stage1_{exp_num}" # to load best model from stage1
        enc_lr = 5e-4
        clf_lr = 1e-3
        
        checkpoint_clf_file = 'best_model_stage1_epoch36.pt'
        classifier_model.load_state_dict(torch.load(os.path.join(checkpoint_clf_dir, checkpoint_clf_file),
                                                    weights_only=True))
        
        checkpoint_clf_dir = f"checkpoints_clf/FORTH_TRACE/w{window_size}/stage2_{exp_num}" # for saving best model in stage2
        classifier_model.unfreeze_last_k_layers(k=top_k)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": classifier_model.context_encoder.parameters(),
                    "lr": enc_lr,
                    "betas": (0.9, 0.99)
                },
                {
                    "params": classifier_model.classifier.parameters(),
                    "lr": clf_lr,
                    "betas": (0.9, 0.99)
                }
            ]
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    history = train_supervised(classifier_model, train_loader, val_loader,
                               optimizer, scheduler, device, max_epochs=max_epochs,
                               checkpoint_dir=checkpoint_clf_dir, patience=patience,stage_two=is_stage_two)

if checkpoint_clf_dir is None:
    if not is_stage_two:
        checkpoint_clf_dir = f"checkpoints_clf/FORTH_TRACE/w{window_size}/stage1_{exp_num}"
    else:
        checkpoint_clf_dir = f"checkpoints_clf/FORTH_TRACE/w{window_size}/stage2_{exp_num}"
    
    classifier_model = JEPAClassifier(context_encoder, 
                                      classifier_head,
                                      freeze_encoder=True).to(device)

logger = setup_logger(checkpoint_clf_dir)

all_files = glob.glob(os.path.join(checkpoint_clf_dir, "*.pt"))
best_checkpoint = all_files[-1]
print(f"Loading best model: {best_checkpoint}")
classifier_model.load_state_dict(torch.load(best_checkpoint, weights_only=True))

test_metrics = evaluate(classifier_model, test_loader, torch.nn.CrossEntropyLoss(), device)
log_evaluation(logger, test_metrics["acc"], test_metrics["conf_matrix"], test_metrics["clf_report"])
print(test_metrics["acc"])
print(test_metrics["conf_matrix"])
print(test_metrics["clf_report"])