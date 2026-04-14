# training/train.py
import os
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm

from training.losses import vicreg_loss
from utils.metrics import representation_stats
from utils.misc import move_to_device
from utils.scheduler import WeightDecayScheduler
from utils.earlystopping import EarlyStoppingSelfSupervised
from utils.logging import setup_logger, log_metrics, log_checkpoint, log_early_stop_progress, log_training_stop

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

def get_cosine_lr_with_warmup(optimizer, num_warmup_steps, num_training_steps, base_lr, max_lr,):
    assert max_lr >= base_lr, "max_lr must be >= base_lr"

    lr_scale = max_lr / base_lr

    def lr_lambda(current_step):
        # Linear warmup: base_lr → max_lr
        if current_step < num_warmup_steps:
            warmup_progress = current_step / float(max(1, num_warmup_steps))
            return 1.0 + warmup_progress * (lr_scale - 1.0)

        # Cosine decay: max_lr → 0
        progress = (current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        return lr_scale * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_cosine_ema_momentum_with_warmup(
    step,
    total_steps,
    warmup_steps=0.1,
    m_start=0.99,
    m_mid=0.996,
    m_end=0.9995,
):
    warmup_steps = int(warmup_steps * total_steps)

    if step < warmup_steps:
        return m_start + (m_mid - m_start) * step / warmup_steps

    t = (step - warmup_steps) / (total_steps - warmup_steps)
    return m_end - (m_end - m_mid) * (
        0.5 * (1 + math.cos(math.pi * t))
    )

def train_self_supervised(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    start_epoch=1,
    max_epochs=100,
    base_lr=1e-4,
    max_lr=1e-3,
    checkpoint_dir="checkpoints",
    checkpoint_freq=5,
    collapse_std_threshold=0.1,
    early_stop_metric='cov',
    patience=20
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = setup_logger(checkpoint_dir)

    early_stopping = EarlyStoppingSelfSupervised(dir_path=checkpoint_dir,
                                                 seq_length=model.seq_length,
                                                 embedding_dim=model.embedding_dim,
                                                 monitor=early_stop_metric, 
                                                 patience=patience, 
                                                 delta=0.0)
    
    # History storage
    history = []
    cnt_early_stop = 0

    total_steps = max_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    lr_scheduler = get_cosine_lr_with_warmup(optimizer, 
                                             num_warmup_steps=warmup_steps, 
                                             num_training_steps=total_steps,
                                             base_lr=base_lr,
                                             max_lr=max_lr)
    
    wd_scheduler = WeightDecayScheduler(optimizer, 
                                        wd_start=0.04, 
                                        wd_end=0.4, 
                                        total_steps=total_steps)

    for epoch in tqdm(range(start_epoch, int(start_epoch + max_epochs)), desc="Training Progress", position=0):
        # Training & Validation
        train_metrics = train_one_epoch(model, train_loader, optimizer, lr_scheduler, wd_scheduler, device, epoch, total_steps)
        val_metrics = validate(model, val_loader, device, epoch)

        log_metrics(logger, epoch, train_metrics, val_metrics)

        # Collapse detection (optional)
        if val_metrics["feat_std"] < collapse_std_threshold:
            cnt_early_stop += 1

            if cnt_early_stop == patience:
                log_early_stop_progress(logger, epoch, cnt_early_stop, patience, "Collapse detected")
                break
        else:
            cnt_early_stop = 0

        # Save checkpoints
        if epoch % checkpoint_freq == 0 or epoch == max_epochs:
            checkpoint_file = f"cpt_epoch{epoch}_w{model.seq_length}_edim{model.embedding_dim}.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            torch.save(model.state_dict(), checkpoint_path)
            log_checkpoint(logger, epoch, checkpoint_path)
        
        else:
            early_stopping(
                epoch=epoch,
                val_cov=val_metrics["cov_pred"],
                val_var=val_metrics["var_pred"],
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

    return history

def train_one_epoch(model, dataloader, optimizer, lr_scheduler, wd_scheduler, device, epoch, total_steps):
    model.train()

    total_loss = 0.0
    total_sim = 0.0
    total_var_pred = 0.0
    total_var_target = 0.0
    total_var_ctx = 0.0      
    total_cov_pred = 0.0
    total_cov_target = 0.0
    total_cov_ctx = 0.0      
    total_std = 0.0
    total_std_ctx = 0.0      
    total_norm = 0.0
    n_batches = 0

    global_step_start = (epoch - 1) * len(dataloader)

    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} train batches", leave=False, position=1)):
        batch = move_to_device(batch, device)

        global_step = global_step_start + step

        z_pred, z_target, z_context = model(batch)

        embed_dim = model.embedding_dim
        loss, loss_dict = vicreg_loss(z_pred, z_target.detach(), embed_dim)

        optimizer.zero_grad()
        loss.backward()

        # Optional but recommended for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        lr_scheduler.step()
        wd_scheduler.step()

        momentum = get_cosine_ema_momentum_with_warmup(
            step=global_step,
            total_steps=total_steps
        )
        model.update_target(momentum)

        # Stats (no grad)
        with torch.no_grad():
            std_pred, norm = representation_stats(z_pred.detach())
            std_ctx, _ = representation_stats(z_context.detach())  # NEW

        total_loss += loss.item()
        total_sim += loss_dict['sim']
        total_var_pred += loss_dict['var_pred']
        total_var_target += loss_dict['var_target']
        #total_var_ctx += loss_dict['var_ctx']
        total_cov_pred += loss_dict['cov_pred']
        total_cov_target += loss_dict['cov_target']
        #total_cov_ctx += loss_dict['cov_ctx']
        total_std += std_pred
        total_std_ctx += std_ctx
        total_norm += norm
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "sim_loss": total_sim / n_batches,
        "var_pred": total_var_pred / n_batches,
        "var_target": total_var_target / n_batches,
        #"var_ctx": total_var_ctx / n_batches,
        "cov_pred": total_cov_pred / n_batches,
        "cov_target": total_cov_target / n_batches,
        #"cov_ctx": total_cov_ctx / n_batches,
        "feat_std": total_std / n_batches,
        "feat_std_ctx": total_std_ctx / n_batches,
        "feat_norm": total_norm / n_batches,
    }


@torch.no_grad()
def validate(model, dataloader, device, epoch):
    model.eval()

    total_loss = 0.0
    total_sim = 0.0
    total_var_pred = 0.0
    total_var_target = 0.0
    total_var_ctx = 0.0
    total_cov_pred = 0.0
    total_cov_target = 0.0
    total_cov_ctx = 0.0
    total_std = 0.0
    total_std_ctx = 0.0
    total_norm = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch} val batches", leave=False, position=1):
        batch = move_to_device(batch, device)

        z_pred, z_target, z_context = model(batch)
        
        embed_dim = model.embedding_dim
        loss, loss_dict = vicreg_loss(z_pred, z_target, embed_dim)

        std_pred, norm = representation_stats(z_pred)
        std_ctx, _ = representation_stats(z_context)

        total_loss += loss.item()
        total_sim += loss_dict['sim']
        total_var_pred += loss_dict['var_pred']
        total_var_target += loss_dict['var_target']
        #total_var_ctx += loss_dict['var_ctx']
        total_cov_pred += loss_dict['cov_pred']
        total_cov_target += loss_dict['cov_target']
        #total_cov_ctx += loss_dict['cov_ctx']
        total_std += std_pred
        total_std_ctx += std_ctx
        total_norm += norm
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "sim_loss": total_sim / n_batches,
        "var_pred": total_var_pred / n_batches,
        "var_target": total_var_target / n_batches,
        #"var_ctx": total_var_ctx / n_batches,
        "cov_pred": total_cov_pred / n_batches,
        "cov_target": total_cov_target / n_batches,
        #"cov_ctx": total_cov_ctx / n_batches,
        "feat_std": total_std / n_batches,
        "feat_std_ctx": total_std_ctx / n_batches,
        "feat_norm": total_norm / n_batches,
    }
