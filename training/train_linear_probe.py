# training/train_clf.py
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from utils.misc import move_to_device
from utils.earlystopping import EarlyStopping
from utils.logging import setup_logger, log_clf_metrics, log_checkpoint

def train_supervised(
        model,
        train_loader,
        val_loader,
        optimizer,
        base_clf_lr,
        device,
        max_epochs=100,
        checkpoint_dir="checkpoints_clf",
        checkpoint_freq=5,
        patience = 10,
        early_stop_metric='acc',
):

    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = setup_logger(checkpoint_dir)

    history = []
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    early_stopping = EarlyStopping(dir_path=checkpoint_dir,
                                   monitor=early_stop_metric, 
                                   patience=patience, 
                                   delta=0.0)

    # Initialize custom LR Scheduler referencing updated optimizer groups dynamically
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    
    for epoch in tqdm(range(1, max_epochs + 1), desc="Training Progress", position=0):
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device, epoch)

        log_clf_metrics(logger, epoch, train_metrics, val_metrics)

        scheduler.step()

        # Save checkpoints
        if epoch % checkpoint_freq == 0 or epoch == max_epochs or epoch >= 70:
            checkpoint_file = f"best_model_progressive_epoch{epoch}.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            torch.save(model.state_dict(), checkpoint_path)
            log_checkpoint(logger, epoch, checkpoint_path)
        else:
            early_stopping(
                epoch=epoch,
                val_metrics=val_metrics,
                model=model
            )

            if early_stopping.early_stop:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch: {early_stopping.best_epoch}"
                )
                break


        history.append({
            "epoch": epoch,
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        # Print discriminative learning rates
        active_lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
        print(
            f"Epoch {epoch:03d} | "
            f"Active LRs: {', '.join(active_lrs)} | "
            f"Tr Loss: {train_metrics['loss']:.3f} | "
            f"Tr Acc: {train_metrics['acc']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.3f} | "
            f"Val Acc: {val_metrics['acc']:.2f}% | "
            f"Val F1: {val_metrics['avg_f1']:.3f}\n"
        )


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(
        dataloader,
        desc=f"Epoch {epoch} train batches",
        leave=False
    ):
        batch = move_to_device(batch, device)
        x = batch["windows"]
        y = batch["labels"][1]       # (B,)

        logits = model(x)        # (B, num_classes)
        
        loss = criterion(logits, y.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        total_loss += loss.item()

    return {
        "loss": total_loss / len(dataloader),
        "acc": 100.0 * correct / total
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    y_pred = []
    y_label = []

    for batch in tqdm(
        dataloader,
        desc=f"Epoch {epoch} val batches",
        leave=False
    ):
        batch = move_to_device(batch, device)

        x = batch["windows"]
        y = batch["labels"][1]       # (B,)

        logits = model(x)        # (B, num_classes)

        loss = criterion(logits, y.long())

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        total_loss += loss.item()

        # Collect results
        y_pred.extend(preds.cpu().numpy())
        y_label.extend(y.cpu().numpy())

    avg_f1 = f1_score(y_label, y_pred, average='macro')

    return {
        "loss": total_loss / len(dataloader),
        "acc": 100.0 * correct / total,
        "avg_f1": avg_f1
    }

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    y_label = np.empty(0,)
    y_pred = np.empty(0,)
    for batch in tqdm(
        dataloader,
        desc=f"test batches",
        leave=False
    ):
        batch = move_to_device(batch, device)

        x = batch["windows"]
        y = batch["labels"][1]       # (B,)

        logits = model(x)        # (B, num_classes)

        loss = criterion(logits, y.long())

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        total_loss += loss.item()
        y_pred = np.append(y_pred, preds.cpu().numpy())
        y_label = np.append(y_label, y.cpu().numpy())

    avg_f1 = f1_score(y_label, y_pred, average='macro')

    return {
        "loss": total_loss / len(dataloader),
        "acc": 100.0 * correct / total,
        "avg_f1": avg_f1,
        "conf_matrix": confusion_matrix(y_label, y_pred),
        "clf_report": classification_report(y_label, y_pred)
    }
