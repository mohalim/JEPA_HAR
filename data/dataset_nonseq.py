import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.dataset import BaseSensorDataset

# JEPA Dataset (Self-Supervised)
class PatchSensorDataset(BaseSensorDataset):
    def __init__(self, root_dir, window_size=100, overlap=0.5, num_patches=5, mask_ratio=0.3, **kwargs):
        super().__init__(root_dir, window_size, overlap, **kwargs)
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        # assert window_size % num_patches == 0, "window_size must be divisible by num_patches"

    def _patchify(self, window):
        return window.reshape(self.num_patches, -1)
        #return window.contiguous().view(self.num_patches, -1)

    def __getitem__(self, idx):
        window = super().__getitem__(idx)
        patches = self._patchify(window)
        
        N = patches.size(0)
        num_mask = max(1, round(self.mask_ratio * N))
        perm = torch.randperm(N)

        masked_idx = perm[:num_mask]
        visible_idx = perm[num_mask:]

        return {
            "visible_patches": patches[visible_idx],    # [Nv, T*C]
            "masked_patches": patches[masked_idx],      # [Nm, T*C]
            "visible_idx": visible_idx,
            "masked_idx": masked_idx,
            "all_patches": patches
        }




# Supervised Dataset (Downstream Task)
class SupervisedSensorDataset(BaseSensorDataset):
    def __getitem__(self, idx):
        subject_id, window_idx = self.index_map[idx]
        window = torch.from_numpy(self.subject_windows[subject_id][window_idx])
        label = self.subject_labels[subject_id][window_idx]
        return window, label


