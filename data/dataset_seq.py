import torch
import numpy as np

from data.dataset import BaseSensorDataset
from data.dataset_nonseq import PatchSensorDataset

# Sequential JEPA Dataset
class SequentialSensorDataset(BaseSensorDataset):
    """
    Returns k sequential windows with no patching or masking.
    Asymmetry is handled in the model or via transforms.
    """
    def __init__(self, root_dir, window_size=100, overlap=0.5, num_windows=2, **kwargs):
        super().__init__(root_dir, window_size, overlap, **kwargs)
        self.num_windows = num_windows

    def __getitem__(self, idx):
        npy_path, start = self.index_map[idx]
        data = self._load_file(npy_path)

        windows = []
        for i in range(self.num_windows):
            s = start + i * self.stride
            window = data[s : s + self.window_size]
            window = self._apply_scaling(window)
            windows.append(torch.from_numpy(window))

        return {
            "windows": windows  # list[k] of (T, C)
        }
    

class SequentialMaskingSensorDataset(BaseSensorDataset):
    """
    Context window is full resolution, target window is masked
    """
    def __init__(
        self,
        root_dir,
        window_size=100,
        overlap=0.5,
        masking_factor=0.5,
        num_windows=2,    
        **kwargs
    ):
        super().__init__(root_dir, window_size, overlap, **kwargs)
        self.masking_factor = masking_factor
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
        for i in range(self.num_windows):
            s = start + i * self.stride
            window = data[s : s + self.window_size]
            window = self._apply_scaling(window)
            window = torch.from_numpy(window)

            windows.append(window)

        mask = self.temporal_block_mask()  # (T,)
        masked_w2 = windows[1].clone()
        masked_w2[~mask] = 0.0   # placeholder; model inserts learned token

        return {
            "w1": windows[0],  # (T, C)
            "w2": windows[1],   # (T, C)
            "masked_w2": masked_w2, # (T, C)
            "mask": mask        # (T,)
        }


class SequentialDownsampledSensorDataset(BaseSensorDataset):
    """
    Context window is downsampled, target window is full resolution.
    """
    def __init__(
        self,
        root_dir,
        window_size=100,
        overlap=0.5,
        downsample_factor=2,
        k=2,
        **kwargs
    ):
        super().__init__(root_dir, window_size, overlap, **kwargs)
        self.downsample_factor = downsample_factor
        self.k = k

    def __getitem__(self, idx):
        npy_path, start = self.index_map[idx]
        data = self._load_file(npy_path)

        windows = []
        for i in range(self.k):
            s = start + i * self.stride
            window = data[s : s + self.window_size]
            window = self._apply_scaling(window)
            window = torch.from_numpy(window)

            windows.append(window)

            if i == 1:
                # downsample window 2
                window_downsampled = window[::self.downsample_factor]
                windows.append(window_downsampled)

            

        return {
            "context": windows[0],  # (T, C)
            "target": windows[1],   # (T, C)
            "target_downsampled": windows[2] # (T/d, C)
        }



# Sequential Patch JEPA Dataset
class SequentialPatchSensorDataset(PatchSensorDataset):
    def __init__(self, root_dir, window_size=100, overlap=0.5, num_patches=5, mask_ratio=0.3, k=2, **kwargs):
        super().__init__(root_dir, window_size, overlap, num_patches, mask_ratio, **kwargs)
        self.k = k              # number of windows in sequence
        # assert len(self.index_map) >= k, "Dataset too small for the given k"

    def __getitem__(self, idx):
        npy_path, start = self.index_map[idx]
        data = self._load_file(npy_path)

        visible_patches, masked_patches, visible_idx, masked_idx, all_patches = [], [], [], [], []

        # Use the same mask indices for both windows
        num_mask = max(1, round(self.mask_ratio * self.num_patches))
        mask_local_idx = torch.randperm(self.num_patches)[:num_mask] # Generate two random values [0-4]

        for i in range(self.k):
            s = start + i * self.stride
            window = data[s : s + self.window_size]  # [T, C]

            patches = (
                torch.as_tensor(window)
                .contiguous()
                .view(self.num_patches, -1)
            )

            all_patches.append(patches)

            # global indices
            offset = i * self.num_patches
            global_idx = torch.arange(self.num_patches) + offset

            mask = torch.zeros(self.num_patches, dtype=torch.bool)
            mask[mask_local_idx] = True

            visible_patches.append(patches[~mask])
            masked_patches.append(patches[mask])

            visible_idx.append(global_idx[~mask])
            masked_idx.append(global_idx[mask])

        return {
            "visible_patches": visible_patches,   # list[k] of (Nv, t*C)
            "masked_patches": masked_patches,     # list[k] of (Nm, t*C)
            "visible_idx": visible_idx,            # list[k] of global indices
            "masked_idx": masked_idx,              # list[k] of global indices
            "all_patches": all_patches,             # list[k] of (N, t*C)
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