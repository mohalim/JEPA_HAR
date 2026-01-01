import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


# Base Dataset Class
class BaseSensorDataset(Dataset):
    def __init__(
        self,
        root_dir,
        window_size=100,
        overlap=0.5,
        scaling="standard",
        eps=1e-8,
        # file_ext="*.csv",
        has_label=True,   # NEW
    ):
        self.root_dir = root_dir
        self.window_size = window_size
        self.stride = int(window_size * (1 - overlap))
        self.scaling = scaling
        self.eps = eps
        # self.file_ext = file_ext
        self.has_label = has_label

        assert self.stride > 0, "Overlap too large, stride becomes zero."

        # Get scaling stats and build index map
        path_stats = os.path.join(root_dir, scaling + "_stats.pkl")
        with open(path_stats, 'rb') as f:
            self.stats = pickle.load(f)
        
        '''
        # Get index mapping
        fname_index_map = f"index_map_w{window_size}_k{num_windows}.pkl"
        with open(os.path.join(root_dir, fname_index_map), 'rb') as f:
            self.index_map = pickle.load(f) # list of (npy_path, start_idx)'''

        self._file_cache = {}  # mmap cache per worker

    def _load_file(self, npy_path):
        if npy_path not in self._file_cache:
            self._file_cache[npy_path] = np.load(npy_path, mmap_mode="r")
        return self._file_cache[npy_path]

    def _apply_scaling(self, signals):
        if self.scaling is None:
            return signals
        if self.scaling == "standard":
            return (signals - self.stats["mean"]) / (self.stats["std"] + self.eps)
        if self.scaling == "minmax":
            return (signals - self.stats["min"]) / (self.stats["max"] - self.stats["min"] + self.eps)
        if self.scaling == "maxabs":
            return signals / (self.stats["maxabs"] + self.eps)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        npy_path, start = self.index_map[idx]
        data = self._load_file(npy_path)
        window = data[start:start + self.window_size]
        window = self._apply_scaling(window)

        return torch.from_numpy(window)


# JEPA Dataset (Self-Supervised)
class PatchSensorDataset(BaseSensorDataset):
    def __init__(self, root_dir, 
                 window_size=100, 
                 overlap=0.5, 
                 num_patches=5, 
                 masking_factor=0.5, 
                 scaling='standard', 
                 **kwargs
                 ):
        super().__init__(root_dir, window_size, overlap, scaling, **kwargs)
        self.num_patches = num_patches
        self.masking_factor = masking_factor
        # assert window_size % num_patches == 0, "window_size must be divisible by num_patches"

    def _patchify(self, window):
        return window.reshape(self.num_patches, -1)

    def __getitem__(self, idx):
        window = super().__getitem__(idx)
        patches = self._patchify(window)
        
        N = patches.size(0)
        num_mask = max(1, round(self.masking_factor * N))
        perm = torch.randperm(N)

        masked_idx = torch.sort(perm[:num_mask])[0]  # sort masked indices
        visible_idx = torch.sort(perm[num_mask:])[0]  # sort visible indices

        return {
            "visible_patches": patches[visible_idx],    # [Nv, T*C]
            "masked_patches": patches[masked_idx],      # [Nm, T*C]
            "visible_idx": visible_idx,
            "masked_idx": masked_idx,
            "all_patches": patches
        }