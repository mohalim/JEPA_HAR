import torch
import os

class EarlyStopping:
    """
    Early stopping for supervised learning.
    Monitors validation loss or accuracy.
    """
    def __init__(
        self,
        dir_path,
        f_stage,
        monitor="loss",      # "loss" or "acc"
        patience=7,
        delta=0.0,
        verbose=False
    ):
        assert monitor in ["loss", "acc"]

        self.dir_path = dir_path
        self.monitor = monitor
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.f_stage = f_stage

        self.counter = 0
        self.early_stop = False
        self.best_epoch = None

        # Initialize best value
        if monitor == "loss":
            self.best = float("inf")
            self.mode = "min"
        else:  # accuracy
            self.best = -float("inf")
            self.mode = "max"

    def __call__(self, epoch, val_loss, val_acc, model):
        """
        Args:
            epoch (int)
            val_loss (float)
            val_acc (float)
            model (nn.Module)
        """
        current = val_loss if self.monitor == "loss" else val_acc

        improved = (
            current < self.best - self.delta
            if self.mode == "min"
            else current > self.best + self.delta
        )

        if improved:
            self.best = current
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(epoch, model)
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping counter: {self.counter}/{self.patience} "
                    f"(best {self.monitor}: {self.best:.4f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, epoch, model):
        """Save best model."""
        if self.verbose:
            print(
                f"Validation {self.monitor} improved. "
                f"Saving model at epoch {epoch}."
            )

        checkpoint_path = os.path.join(
            self.dir_path, f"best_model_{self.f_stage}_epoch{epoch}.pt"
        )
        torch.save(model.state_dict(), checkpoint_path)


class EarlyStoppingSelfSupervised:
    def __init__(
        self,
        dir_path,
        seq_length,
        embedding_dim,
        monitor="cov",      # "cov" or "var"
        patience=7,
        delta=0.0,
        verbose=False
    ):
        assert monitor in ["cov", "var"]

        self.dir_path = dir_path
        self.monitor = monitor
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.seq_length = seq_length
        self.embedding_dim = embedding_dim

        self.counter = 0
        self.early_stop = False
        self.best_epoch = None 

        # Initialize best value
        if monitor == "cov":
            self.best = float("inf")
            self.mode = "min"
        else:  # variance
            self.best = -float("inf")
            self.mode = "max"

    def __call__(self, epoch, val_cov, val_var, model):

        current = val_cov if self.monitor == "cov" else val_var

        improved = (
            current < self.best - self.delta
            if self.mode == "min"
            else current > self.best + self.delta
        )

        if improved:
            self.best = current
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(epoch, model)
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping counter: {self.counter}/{self.patience} "
                    f"(best {self.monitor}: {self.best:.4f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, epoch, model):
        """Save best model."""
        if self.verbose:
            print(
                f"Validation {self.monitor} improved. "
                f"Saving model at epoch {epoch}."
            )

        checkpoint_path = os.path.join(
            self.dir_path, f"cpt_epoch{epoch}_w{self.seq_length}_edim{self.embedding_dim}.pt"
            
        )
        torch.save(model.state_dict(), checkpoint_path)