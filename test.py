from data.dataset import SequentialJEPASensorDataset
from torch.utils.data import DataLoader

dataset = SequentialJEPASensorDataset(
    root_dir="data/SBRHAPT/Train/",
    window_size=100,
    overlap=0.5,
    num_patches=5,
    mask_ratio=0.3
)

loader = DataLoader(dataset, batch_size=10, shuffle=True)

for data in loader:
    v1,v2 = data["visible_patches"]
    v1_idx, v2_idx = data["visible_idx"]
    m1, m2 = data["masked_patches"]
    m1_idx, m2_idx = data["masked_idx"]

    print(v1_idx, v2_idx)
    print(m1_idx, m2_idx)

    break