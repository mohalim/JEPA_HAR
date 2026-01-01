from data.dataset_seq import SequentialPatchSensorDataset
from torch.utils.data import DataLoader

dataset = SequentialPatchSensorDataset(
    root_dir="../../Datasets/REALDISP_AccGyro/train_npy/",
    window_size=100,
    overlap=0.5,
    num_patches=5,
    masking_factor=0.5
)

loader = DataLoader(dataset, batch_size=10, shuffle=True)

for data in loader:
    w1,w2 = data["windows"]
    v1_idx, v2_idx = data["visible_idx"]
    m1_idx, m2_idx = data["masked_idx"]
    print(w1.shape)
    print(w2.shape)
    print(v1_idx, v2_idx)
    print(m1_idx, m2_idx)

    break