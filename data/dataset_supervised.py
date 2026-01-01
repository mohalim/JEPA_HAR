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


