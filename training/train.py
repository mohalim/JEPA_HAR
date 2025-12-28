# training/train.py
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from training.losses import vicreg_loss
from utils.metrics import representation_stats
from utils.misc import move_to_device

# If feature std collapses, your model is dead — even if loss improves.
# feat_std
# Collapse = all features identical → std → 0 e.g. Collapsing → 0.0 – 0.2
# Rises quickly in early epochs, then stabilizes in a range e.g. JEPA / BYOL ~0.6 – 1.2, VICReg ~0.8 – 1.5

# feat_norm
# Average L2 norm per sample
# Norm → 0 → trivial collapse, Norm → ∞ → instability
# Stable over time, no explosions, no vanishing
# After normalization: ~1.0

# Loss plateaus early → normal
# Validation loss not decreasing → normal
# Train > Val loss → normal
def train_self_supervised(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    momentum=0.995,
    start_epoch=1,
    max_epochs=100,
    checkpoint_dir="checkpoints",
    checkpoint_freq=5,
    collapse_std_threshold=0.1,
    patience = 10
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    # History storage
    history = []
    cnt_early_stop = 0

    # Optional: sanity early stopping (collapse detection)
    # early_stopping = EarlyStopping(patience=200, delta=0.0, path=os.path.join(checkpoint_dir, "best_model.pt"))
    max_epochs = int(start_epoch + max_epochs)
    for epoch in tqdm(range(start_epoch, max_epochs), desc="Training Progress", position=0):
        # Training & Validation
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, momentum, epoch)
        val_metrics = validate(model, val_loader, device, epoch)

        scheduler.step()

        # Collapse detection (optional)
        if train_metrics["feat_std"] < collapse_std_threshold:
            cnt_early_stop += 1

            if cnt_early_stop == patience:
                print(f"Collapse detected at epoch {epoch}. Stopping training.")
                break
        else:
            cnt_early_stop = 0

        # Save checkpoints
        if epoch % checkpoint_freq == 0 or epoch == max_epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        history.append({
            "epoch": epoch,
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        print(f"\nEpoch {epoch:03d} | "
          f"Tr Loss: {train_metrics['loss']:.3f} | "
          f"Tr Sim Loss: {train_metrics['sim_loss']:.3f} | "
          f"Tr Var Loss: {train_metrics['var_loss']:.3f} | "
          # f"Tr Cov Loss: {train_metrics['cov_loss']:.3f} | "
          f"Tr Std: {train_metrics['feat_std']:.3f} | "
          f"Tr Norm: {train_metrics['feat_norm']:.3f} | "
          f"Val Loss: {val_metrics['loss']:.3f} | "
          f"Val Sim Loss: {val_metrics['sim_loss']:.3f} | "
          f"Val Var Loss: {val_metrics['var_loss']:.3f} | "
          # f"Val Cov Loss: {val_metrics['cov_loss']:.3f} | "
          f"Val Std: {val_metrics['feat_std']:.3f} | "
          f"Val Norm: {val_metrics['feat_norm']:.3f}")

        '''
        # Check early stopping condition
        early_stopping(val_metrics["loss"], model)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break'''

    return history

def train_one_epoch(model, dataloader, optimizer, device, momentum, epoch):
    model.train()

    total_loss = 0.0
    total_sim = 0.0
    total_var = 0.0
    total_cov = 0.0
    total_std = 0.0
    total_norm = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch} train batches", leave=False, position=1):
        batch = move_to_device(batch, device)

        z_pred, z_target = model(batch)

        loss, sim_loss, var_loss = vicreg_loss(z_pred, z_target.detach())

        optimizer.zero_grad()
        loss.backward()
        # Optional but recommended for stability
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.update_target(momentum)

        # Stats (no grad)
        with torch.no_grad():
            std, norm = representation_stats(z_pred.detach())

        total_loss += loss.item()
        total_sim += sim_loss.item()
        total_var += var_loss.item()
        # total_cov += cov_loss.item()
        total_std += std
        total_norm += norm
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "sim_loss": total_sim / n_batches,
        "var_loss": total_var / n_batches,
        # "cov_loss": total_cov / n_batches,
        "feat_std": total_std / n_batches,
        "feat_norm": total_norm / n_batches,
    }


@torch.no_grad()
def validate(model, dataloader, device, epoch):
    model.eval()

    total_loss = 0.0
    total_sim = 0.0
    total_var = 0.0
    total_cov = 0.0
    total_std = 0.0
    total_norm = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch} val batches", leave=False, position=1):
        batch = move_to_device(batch, device)

        z_pred, z_target = model(batch)

        loss, sim_loss, var_loss = vicreg_loss(z_pred, z_target)

        std, norm = representation_stats(z_pred)

        total_loss += loss.item()
        total_sim += sim_loss.item()
        total_var += var_loss.item()
        # total_cov += cov_loss.item()
        total_std += std
        total_norm += norm
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "sim_loss": total_sim / n_batches,
        "var_loss": total_var / n_batches,
        # "cov_loss": total_cov / n_batches,
        "feat_std": total_std / n_batches,
        "feat_norm": total_norm / n_batches,
    }
