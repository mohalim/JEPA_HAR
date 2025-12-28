# main.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.jepa import JEPA
from models.classifier import JEPAClassifier
from training.train import train_one_epoch
from data.dataset import JEPASensorDataset, SupervisedSensorDataset
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

lr = 5e-5
momentum = 0.95
batch_size = 64

# model's parameters
window_size = 100
num_patches = 5
channels = 6
patch_dim = (window_size // num_patches) * channels
embedding_dim = 128
hidden_dim = 256

# -----------------------------
# 1. JEPA self-supervised pretraining
# -----------------------------
model = JEPA(
    patch_dim=patch_dim, embedding_dim=embedding_dim, 
    num_patches=num_patches, hidden_dim=hidden_dim
    ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)


dataset = JEPASensorDataset(
    root_dir="data/SBRHAPT/Train/",
    window_size=100,
    overlap=0.5,
    num_patches=5,
    mask_ratio=0.3
)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

epochs = 60
losses = []
for epoch in range(epochs):
    loss = train_one_epoch(model, loader, optimizer, device, momentum)
    losses.append(loss)
    print(f"Epoch {epoch}: Loss={loss:.4f}")


# -----------------------------
# 2. Activity classifier supervised training
# -----------------------------
supervised_dataset = SupervisedSensorDataset(
    root_dir="data/SBRHAPT/Test/",
    window_size=window_size,
    overlap=0.5
)
supervised_loader = DataLoader(supervised_dataset, batch_size=batch_size, shuffle=True)

num_classes = 12  # adjust to your dataset
classifier_model = JEPAClassifier(model, embedding_dim=embedding_dim, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier_model.parameters(), lr=1e-3)

num_epochs = 60
for epoch in range(num_epochs):
    classifier_model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for windows, labels in supervised_loader:
        windows = windows.to(device, dtype=torch.float)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = classifier_model(windows)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * windows.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += windows.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.4f}")

#plt.plot(np.arange(1, 101, 1), losses)
#plt.show()

# for debugging
'''
from models.encoder import PatchTransformerEncoder
import torch.nn as nn
context_encoder = PatchTransformerEncoder(patch_dim, embedding_dim, num_patches)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
for batch in dataloader:
    patch_data = batch['visible_patches']
    patch_idx = batch['visible_idx']
    #context_encoder(patch_data, patch_idx[0])

    pos_embed = nn.Parameter(torch.randn(1, num_patches, 3))
    print(pos_embed)
    print(pos_embed[:, patch_idx[0], :])
    break
'''

