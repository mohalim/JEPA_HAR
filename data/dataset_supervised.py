import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.dataset import BaseSensorDataset

# Supervised Dataset (Downstream Task)
class SupervisedSensorDataset(BaseSensorDataset):
    def __getitem__(self, idx):
        subject_id, window_idx = self.index_map[idx]
        window = torch.from_numpy(self.subject_windows[subject_id][window_idx])
        label = self.subject_labels[subject_id][window_idx]
        return window, label


class SequentialSupervisedSensorDataset(BaseSensorDataset):
    def __init__(
        self,
        root_dir,
        window_size=102,
        overlap=0.5,
        num_windows=2,    
        **kwargs
    ):
        super().__init__(root_dir, window_size, overlap, **kwargs)
        self.num_windows = num_windows

        # Get index mapping
        fname_index_map = f"index_map_w{window_size}_k{num_windows}.pkl"
        with open(os.path.join(root_dir, fname_index_map), 'rb') as f:
            self.index_map = pickle.load(f) # list of (npy_path, start_idx)

    def __getitem__(self, idx):
        npy_path, start = self.index_map[idx]
        data = self._load_file(npy_path)

        windows = []
        labels = []
        for i in range(self.num_windows):
            s = start + i * self.stride

            # Features only, last column is label
            window = data[s : s + self.window_size, :-1]
            window = self._apply_scaling(window)
                        
            x = torch.from_numpy(window)  # (T, C)
            windows.append(x)

            label = torch.from_numpy(data[s : s + self.window_size, -1])
            y = torch.mode(label).values - 1
            labels.append(y)
        
        return {
            "windows": windows,  # [(T, C), (T, C)]
            "labels": labels  # (num_windows,)
        }

