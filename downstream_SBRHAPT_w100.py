import torch
import os
import glob
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings

from models.jepa_pos import JEPA_SEQ
from models.classifier import JEPAClassifier
from data.dataset_seq import SequentialSupervisedSensorDataset
from training.train_clf import train_supervised, evaluate

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window_size = 100
batch_size = 64
channels = 6

train_path_dataset = "../../Datasets/SBRHAPT/train_npy/"
val_path_dataset = "../../Datasets/SBRHAPT/val_npy/"
test_path_dataset = "../../Datasets/SBRHAPT/test_npy/"

train_dataset = SequentialSupervisedSensorDataset(
    root_dir=train_path_dataset,
    window_size=window_size,
    overlap=0.5,
)

val_dataset = SequentialSupervisedSensorDataset(
    root_dir=val_path_dataset,
    window_size=window_size,
    overlap=0.5,
)

test_dataset = SequentialSupervisedSensorDataset(
    root_dir=test_path_dataset,
    window_size=window_size,
    overlap=0.5,
)

train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=4,
                          persistent_workers=True,
                          pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

embedding_dim = 256
hidden_dim = 384
model = JEPA_SEQ(
    input_channels=channels, embedding_dim=embedding_dim, 
    window_size=window_size, hidden_dim=hidden_dim
    )

checkpoint_dir = 'checkpoints'
checkpoint_file = 'checkpoint_epoch_100.pt'
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint_file), weights_only=True))
model.eval()
context_encoder = model.context_encoder

num_classes = 12
classifier_head = nn.Sequential(
            nn.Linear(embedding_dim, int(embedding_dim/2)),
            nn.GELU(),
            nn.Linear(int(embedding_dim/2), num_classes)
            )
classifier_model = JEPAClassifier(context_encoder, 
                                  classifier_head, 
                                  embedding_dim=embedding_dim, 
                                  num_classes=num_classes).to(device)


is_train = False
checkpoint_clf_dir = f"checkpoints_clf/w{window_size}"
if is_train:
    max_epochs = 100
    lr = 2e-3
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    history = train_supervised(classifier_model, train_loader, val_loader,
                               optimizer, scheduler, device, max_epochs=max_epochs,
                               checkpoint_dir=checkpoint_clf_dir, patience = 10)

all_files = glob.glob(os.path.join(checkpoint_clf_dir, "*.pt"))
best_checkpoint = sorted(all_files)[-1]
classifier_model.load_state_dict(torch.load(best_checkpoint, weights_only=True))

test_metrics = evaluate(classifier_model, test_loader, torch.nn.CrossEntropyLoss(), device)
print(test_metrics["acc"])
print(test_metrics["conf_matrix"])
print(test_metrics["clf_report"])