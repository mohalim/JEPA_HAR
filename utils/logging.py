import logging
import os

from tqdm import tqdm

def setup_logger(log_dir, filename="training.log", name="train"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicate logs

    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler (optional, tqdm-safe)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

import numpy as np

def log_evaluation(logger, accuracy, conf_matrix, class_report, epoch=None):
    """
    Logs accuracy, confusion matrix, and classification report.
    
    Args:
        logger: The logger instance from setup_logger.
        accuracy: float, the accuracy score.
        conf_matrix: array-like, the confusion matrix.
        class_report: str, the classification report string from sklearn.
        epoch: int (optional), the current training epoch.
    """
    header = f" METRICS - EPOCH {epoch} " if epoch is not None else " METRICS "
    
    logger.info(f"{'='*20}{header}{'='*20}")
    
    # Log Accuracy
    logger.info(f"Overall Accuracy: {accuracy:.4f}%")
    
    # Log Confusion Matrix 
    # Using np.array2string to maintain formatting in the text log
    matrix_str = np.array2string(np.array(conf_matrix), separator=', ')
    logger.info(f"Confusion Matrix:\n{matrix_str}")
    
    # Log Classification Report
    logger.info(f"Classification Report:\n{class_report}")
    
    logger.info(f"{'='*50}\n")

def log_metrics(logger, epoch, train_metrics, val_metrics):
    msg = (
        f"Epoch {epoch:03d}\n"
        f"Tr Loss {train_metrics['loss']:.3f}, "
        f"Sim {train_metrics['sim_loss']:.3f}, "
        f"Var Pd {train_metrics['var_pred']:.3f}, "
        #f"Var Cx {train_metrics['var_ctx']:.3f}, "
        f"Cov Pd {train_metrics['cov_pred']:.3f}, "
        #f"Cov Cx {train_metrics['cov_ctx']:.3f}, "
        f"Std Pd {train_metrics['feat_std']:.3f}, "
        f"Std Cx {train_metrics['feat_std_ctx']:.3f}, "
        f"Norm Pd {train_metrics['feat_norm']:.3f}\n"
        f"Val Loss {val_metrics['loss']:.3f}, "
        f"Sim {val_metrics['sim_loss']:.3f}, "
        f"Var Pd {val_metrics['var_pred']:.3f}, "
        #f"Var Cx {val_metrics['var_ctx']:.3f}, "
        f"Cov Pd {val_metrics['cov_pred']:.3f}, "
        #f"Cov Cx {train_metrics['cov_ctx']:.3f}, "
        f"Std Pd {val_metrics['feat_std']:.3f}, "
        f"Std Cx {val_metrics['feat_std_ctx']:.3f}, "
        f"Norm Pd {val_metrics['feat_norm']:.3f}\n"
    )

    logger.info(msg)      # clean, single-line in file
    #tqdm.write(msg)       # clean console output

def log_clf_metrics(logger, epoch, train_metrics, val_metrics):
    msg = (
        f"Epoch {epoch:03d}\n"
        f"Tr Loss {train_metrics['loss']:.3f}, "
        f"Tr Acc {train_metrics['acc']:.3f}, "
        f"Val Loss {val_metrics['loss']:.3f}, "
        f"Val Acc {val_metrics['acc']:.3f}\n"
    )

    logger.info(msg)      # clean, single-line in file

def log_checkpoint(logger, epoch, path):
    msg = (
        f"Checkpoint saved | epoch={epoch} | "
        f"path={path}"
    )
    logger.info(msg)
    #tqdm.write(msg)


def log_early_stop_progress(logger, epoch, cnt, patience, reason):
    msg = (
        f"Early-stop counter {cnt}/{patience} at epoch {epoch} "
        f"(reason: {reason})"
    )
    logger.warning(msg)
    tqdm.write(msg)


def log_training_stop(logger, epoch, reason):
    msg = f"Training stopped at epoch {epoch} | reason: {reason}"
    logger.error(msg)
    tqdm.write(msg)
