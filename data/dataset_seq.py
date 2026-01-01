import torch
import os
import pickle

from data.dataset import BaseSensorDataset


# Sequential JEPA Dataset
class SequentialSensorDataset(BaseSensorDataset):
    def __init__(self, root_dir, 
                 window_size=102, 
                 overlap=0.5,
                 patch_size=17,
                 num_windows=2,
                 masking_factor=0.5,
                 **kwargs):
        super().__init__(root_dir, window_size, overlap, **kwargs)
        self.num_windows = num_windows         # number of windows in sequence
        self.num_patches = int(window_size / patch_size)
        self.masking_factor = masking_factor
        
        # Get index mapping
        fname_index_map = f"index_map_w{window_size}_k{num_windows}.pkl"
        with open(os.path.join(root_dir, fname_index_map), 'rb') as f:
            self.index_map = pickle.load(f) # list of (npy_path, start_idx)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        npy_path, start = self.index_map[idx]
        data = self._load_file(npy_path)

        windows, enc_mask_idx, pred_mask_idx = [], [], []

        # Use the same mask indices for both windows
        num_mask = max(1, round(self.masking_factor * self.num_patches))
        mask_local_idx = torch.randperm(self.num_patches)[:num_mask] # Generate two random values
        mask_local_idx, _ = torch.sort(mask_local_idx)  # ensure ordered masking

        for i in range(self.num_windows):
            s = start + i * self.stride
            window = data[s : s + self.window_size]  # [T, C]
            
            windows.append(window)
            global_idx = torch.arange(self.num_patches) # + offset

            mask = torch.zeros(self.num_patches, dtype=torch.bool)
            mask[mask_local_idx] = True

            enc_mask_idx.append(global_idx[~mask])
            pred_mask_idx.append(global_idx[mask])

        return {
            "windows": windows,                     # list[k] of (L, C)
            "enc_mask_idx": enc_mask_idx,           # list[k] of (ei,) - visible patch indices
            "pred_mask_idx": pred_mask_idx,         # list[k] of (pi,) - masked patch indices
        }

# =======================
# Sequential Supervised Dataset
# =======================
class SequentialSupervisedSensorDataset(BaseSensorDataset):
    def __init__(
        self,
        root_dir,
        window_size=100,
        overlap=0.5,
        num_windows=2,    
        **kwargs
    ):
        super().__init__(root_dir, window_size, overlap, num_windows, **kwargs)
        self.num_windows = num_windows

    def temporal_block_mask(self):
        """
        Returns a boolean mask of shape (T,)
        True = keep token
        False = mask token
        """
        num_mask = int(self.window_size * self.masking_factor)
        start = torch.randint(0, self.window_size - num_mask + 1, (1,)).item()
        mask = torch.ones(self.window_size, dtype=torch.bool)
        mask[start : start + num_mask] = False
        return mask

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
            "w1": windows[0],  # (T, C)
            "w2": windows[1],  # (T, C)
            "labels": labels  # (num_windows,)
        }


'''
# Sequential JEPA Dataset
class SequentialSensorDataset(BaseSensorDataset):
    """
    Returns k sequential windows with no patching or masking.
    Asymmetry is handled in the model or via transforms.
    """
    def __init__(self,
        root_dir,
        window_size=100,
        overlap=0.5,
        masking_factor=0.5,
        reverse_ratio=0.5,
        noise_std=0.01,
        num_windows=2,    
        **kwargs):
        super().__init__(root_dir, window_size, overlap, **kwargs)
        self.num_windows = num_windows
        self.masking_factor = masking_factor
        self.reverse_ratio = reverse_ratio
        self.noise_std = noise_std

    def temporal_block_mask(self):
        num_mask = int(self.window_size * self.masking_factor)
        start = torch.randint(0, self.window_size - num_mask + 1, (1,)).item()
        mask = torch.ones(self.window_size, dtype=torch.bool)
        mask[start : start + num_mask] = False      # False index indicates samples will be masked
        return mask

    def apply_channel_noise(self, window):
        noise = torch.randn_like(window) * self.noise_std
        return window + noise

    def permute_channels(self, window):
        C = window.size(1)
        perm_idx = torch.randperm(C)
        return window[:, perm_idx]
    
    def reverse_channels(self, window):
        C = window.size(1)
        num_reverse = int(C * self.reverse_ratio)
        if num_reverse == 0:
            return window
        channels_to_reverse = torch.randperm(C)[:num_reverse]
        window[:, channels_to_reverse] = window[:, channels_to_reverse].flip(0)
        return window

    def __getitem__(self, idx):
        npy_path, start = self.index_map[idx]
        data = self._load_file(npy_path)

        # Load raw windows
        windows = []
        for i in range(self.num_windows):
            s = start + i * self.stride
            window = data[s : s + self.window_size]
            window = self._apply_scaling(window)
            window = torch.from_numpy(window)
            windows.append(window)

        # ----------------- w1 (context) -----------------
        w1 = windows[0]  # clean context window

        # ----------------- w2 (target) -----------------
        w2 = windows[1]  # clean target for target encoder

        # ----------------- Apply augmentations BEFORE masking -----------------
        masked_w2 = w2.clone()
        masked_w2 = self.apply_channel_noise(masked_w2)
        masked_w2 = self.reverse_channels(masked_w2)
        masked_w2 = self.permute_channels(masked_w2)

        # ----------------- Mask target window -----------------
        mask = self.temporal_block_mask()  # (T,)
        #masked_w2 = w2_aug.clone()
        masked_w2[~mask] = 0.0  # will be replaced by learned mask token in model

        return {
            "w1": w1,                       # (T, C) clean context window
            "w2": w2,                       # (T, C) clean target for target encoder
            "masked_w2": masked_w2,         # (T, C) masked + augmented
            "mask": mask                     # (T,)
        }'''
    

