''' Not used '''
''' For debugging '''

from data.dataset_seq import SequentialSensorDataset
from torch.utils.data import DataLoader

batch_size = 4
window_size = 102
num_windows = 2
channels = 6
patch_size = 17

train_path_dataset = "../../Datasets/REALDISP_AccGyro/train_npy/"
val_path_dataset = "../../Datasets/REALDISP_AccGyro/val_npy/"

train_dataset = SequentialSensorDataset(
    root_dir=train_path_dataset,
    window_size=window_size,
    overlap=0.5,
    patch_size=patch_size,
    num_windows=num_windows,
)

val_dataset = SequentialSensorDataset(
    root_dir=val_path_dataset,
    window_size=window_size,
    overlap=0.5,
    patch_size=patch_size,
    num_windows=num_windows,
)

train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=False, 
                          num_workers=4,
                          persistent_workers=True,
                          pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for data in train_loader:

    w1,w2 = data["windows"]

    break