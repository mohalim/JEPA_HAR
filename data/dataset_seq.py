import torch
import os
import pickle

from data.dataset import BaseSensorDataset

# Sequential JEPA Dataset
class SequentialSensorDataset(BaseSensorDataset):
    '''
    This dataset returns indices of visible and masked patches
    '''
    def __init__(self, root_dir, 
                 window_size=102, 
                 overlap=0.1,
                 patch_size=17,
                 num_windows=2,
                 masking_factor=0.5,
                 reverse_ratio=0.5,
                 noise_std=0.01,
                 **kwargs):
        super().__init__(root_dir, window_size, overlap, **kwargs)
        self.num_windows = num_windows         # number of windows in sequence
        self.num_patches = int(window_size / patch_size)
        self.patch_size = patch_size
        self.masking_factor = masking_factor
        self.reverse_ratio = reverse_ratio
        self.noise_std = noise_std
        
        # Get index mapping
        fname_index_map = f"index_map_w{window_size}_k{num_windows}_ov{int(overlap*100)}.pkl"
        with open(os.path.join(root_dir, fname_index_map), 'rb') as f:
            self.index_map = pickle.load(f) # list of (npy_path, start_idx)

    def apply_channel_noise(self, window):
        noise = torch.randn_like(window) * self.noise_std
        return window + noise
    
    def reverse_channels(self, window):
        window = window.clone()
        C = window.size(1)
        num_reverse = int(C * self.reverse_ratio)
        if num_reverse == 0:
            return window
        channels_to_reverse = torch.randperm(C)[:num_reverse]
        window[:, channels_to_reverse] = window[:, channels_to_reverse].flip(0)
        return window

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        npy_path, start = self.index_map[idx]
        data = self._load_file(npy_path)

        # windows = []
        windows, enc_mask_idx, pred_mask_idx = [], [], []
        
        # Use the same mask indices for both windows
        num_mask = max(1, round(self.masking_factor * self.num_patches))      

        for i in range(self.num_windows):
            s = start + i * self.stride
            window = data[s : s + self.window_size]  # [T, C]
            window = torch.from_numpy(window)
            # Augmentation 1: noise additive
            is_apply_noise = torch.randint(low=0, high=2, size=(1,))
            if is_apply_noise:
                window = self.apply_channel_noise(window)
            # Augmentation 2: channel reversal
            window = self.reverse_channels(window)

            windows.append(window)

            # offset = i * self.num_patches
            global_idx = torch.arange(self.num_patches) #+ offset

            mask_local_idx = torch.randperm(self.num_patches)[:num_mask] # Generate random values for masking
            mask_local_idx, _ = torch.sort(mask_local_idx)  # ensure ordered masking
            mask = torch.zeros(self.num_patches, dtype=torch.bool)
            mask[mask_local_idx] = True

            enc_mask_idx.append(global_idx[~mask])
            pred_mask_idx.append(global_idx[mask])
        
        return {
            "windows": windows,                     # list[k] of (L, C)
            "enc_mask_idx": enc_mask_idx,           # (ei,) - visible patch indices
            "pred_mask_idx": pred_mask_idx,         # (pi,) - masked patch indices
        }

''' Not used '''
class SequentialSensorChannelDataset(BaseSensorDataset):
    '''
    This dataset returns indices of visible and masked channels
    '''
    def __init__(self, root_dir, 
                 window_size=102, 
                 overlap=0.1,
                 num_channels=6,
                 num_windows=2,
                 masking_factor=0.5,
                 **kwargs):
        super().__init__(root_dir, window_size, overlap, **kwargs)
        self.num_channels = num_channels
        self.num_windows = num_windows         # number of windows in sequence
        self.masking_factor = masking_factor
        
        # Get index mapping
        fname_index_map = f"index_map_w{window_size}_k{num_windows}_ov{int(overlap*100)}.pkl"
        with open(os.path.join(root_dir, fname_index_map), 'rb') as f:
            self.index_map = pickle.load(f) # list of (npy_path, start_idx)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        npy_path, start = self.index_map[idx]
        data = self._load_file(npy_path)

        # windows = []
        windows, enc_mask_idx, pred_mask_idx = [], [], []
        
        # Use the same mask indices for both windows
        num_mask = max(1, round(self.masking_factor * self.num_channels))      

        for i in range(self.num_windows):
            s = start + i * self.stride
            window = data[s : s + self.window_size]  # [T, C]
            windows.append(window)

            # offset = i * self.num_patches
            global_idx = torch.arange(self.num_channels) #+ offset

            mask_local_idx = torch.randperm(self.num_channels)[:num_mask] # Generate random values for masking
            mask_local_idx, _ = torch.sort(mask_local_idx)  # ensure ordered masking
            mask = torch.zeros(self.num_channels, dtype=torch.bool)
            mask[mask_local_idx] = True

            enc_mask_idx.append(global_idx[~mask])
            pred_mask_idx.append(global_idx[mask])
        
        return {
            "windows": windows,                     # list[k] of (L, C)
            "enc_mask_idx": enc_mask_idx,           # (ei,) - visible patch indices
            "pred_mask_idx": pred_mask_idx,         # (pi,) - masked patch indices
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
    

