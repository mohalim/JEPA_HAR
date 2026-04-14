# training/train_clf.py
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from utils.misc import move_to_device
from utils.earlystopping import EarlyStopping
from utils.logging import setup_logger, log_clf_metrics, log_checkpoint

def train_supervised(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        max_epochs=100,
        checkpoint_dir="checkpoints_clf",
        checkpoint_freq=5,
        patience = 10,
        early_stop_metric='acc',
        stage_two=False
):
    
    if not stage_two:
        f_stage = "stage1"
    else:
        f_stage = "stage2"

    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = setup_logger(checkpoint_dir)

    history = []
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    early_stopping = EarlyStopping(dir_path=checkpoint_dir,
                                   f_stage=f_stage,
                                   monitor=early_stop_metric, 
                                   patience=patience, 
                                   delta=0.0)
    
    for epoch in tqdm(range(1, max_epochs + 1), desc="Training Progress", position=0):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device, epoch)

        log_clf_metrics(logger, epoch, train_metrics, val_metrics)

        scheduler.step()

        # Save checkpoints
        if epoch % checkpoint_freq == 0 or epoch == max_epochs:
            checkpoint_file = f"best_model_{f_stage}_epoch{epoch}.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            torch.save(model.state_dict(), checkpoint_path)
            log_checkpoint(logger, epoch, checkpoint_path)
        else:
            early_stopping(
                epoch=epoch,
                val_loss=val_metrics["loss"],
                val_acc=val_metrics["acc"],
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

        print(
            f"Epoch {epoch:03d} | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | "
            f"Tr Loss: {train_metrics['loss']:.3f} | "
            f"Tr Acc: {train_metrics['acc']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.3f} | "
            f"Val Acc: {val_metrics['acc']:.2f}%\n"
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

    return {
        "loss": total_loss / len(dataloader),
        "acc": 100.0 * correct / total
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

    return {
        "loss": total_loss / len(dataloader),
        "acc": 100.0 * correct / total,
        "conf_matrix": confusion_matrix(y_label, y_pred),
        "clf_report": classification_report(y_label, y_pred)
    }
