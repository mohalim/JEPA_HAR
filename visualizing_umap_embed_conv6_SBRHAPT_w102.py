import torch
import os
import glob
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings
import umap
import matplotlib.pyplot as plt

from models.classifier_conv import JEPAClassifier
from models.jepa_conv import JEPA_SEQ
from data.dataset_supervised import SequentialSupervisedSensorDataset
from utils.misc import extract_embeddings

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

is_winseq = True
top_k = 4       # number of final layers of encoder to fine-tune

exp_num = "conv6_e440"  # conv6_e440
is_stage_two = False

window_size = 102
overlap = 0.5
num_windows = 2
channels = 6
patch_size = 6      # window_size / patch_size = 17 patches
batch_size = 64
     
test_path_dataset = "../../Datasets/SBRHAPT/test_npy/"

test_dataset = SequentialSupervisedSensorDataset(
    root_dir=test_path_dataset,
    window_size=window_size,
    overlap=overlap,
    num_windows=num_windows
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if not is_winseq:
    checkpoint_dir = f"checkpoints/w{window_size}/no_seq"
else:
    checkpoint_dir = f"checkpoints/w{window_size}/win_seq_{exp_num}"

kernel_sizes = [7, 5, 3, 3]
embedding_dim = 440
predictor_embed_dim = 220
predictor_n_heads = 4
predictor_n_layers = 2

model = JEPA_SEQ(
    seq_length=window_size,
    channels=channels,
    patch_size=patch_size,
    conv_kernel_sizes=kernel_sizes,
    #num_windows=num_windows, 
    embedding_dim=embedding_dim,
    predictor_embed_dim=predictor_embed_dim,
    predictor_n_heads=predictor_n_heads,
    predictor_n_layers=predictor_n_layers,
    is_seq=is_winseq
    ).to(device)

checkpoint_file = 'cpt_epoch100_w102_edim440.pt'
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint_file), weights_only=True))
model.eval()
context_encoder = model.context_encoder

num_classes = 12
classifier_head = nn.Sequential(
            nn.Linear(embedding_dim, int(embedding_dim/2)),
            nn.GELU(),
            #nn.LayerNorm(int(embedding_dim/2)),
            nn.Dropout(0.1),
            nn.Linear(int(embedding_dim/2), num_classes)
            )

classifier_model = JEPAClassifier(context_encoder, 
                                  classifier_head,
                                  freeze_encoder=True).to(device)

if not is_stage_two:
    checkpoint_clf_dir = f"checkpoints_clf/SBHARPT/w{window_size}/stage1_{exp_num}"
    visualization_dir = f"visualization/SBHARPT/w{window_size}/stage1_{exp_num}"
else:
    checkpoint_clf_dir = f"checkpoints_clf/SBHARPT/w{window_size}/stage2_{exp_num}"
    visualization_dir = f"visualization/SBHARPT/w{window_size}/stage2_{exp_num}"

os.makedirs(visualization_dir, exist_ok=True)

stageNo = 1 if not is_stage_two else 2
best_checkpoint = f'best_model_stage{stageNo}_epoch90.pt'
path_checkpoint = os.path.join(checkpoint_clf_dir, best_checkpoint)

#all_files = glob.glob(os.path.join(checkpoint_clf_dir, "*.pt"))
#best_checkpoint = all_files[-1]
print(f"Loading best model: {path_checkpoint}")
classifier_model.load_state_dict(torch.load(path_checkpoint, weights_only=True))


embeddings, labels = extract_embeddings(
    classifier_model, test_loader, device, dim=1
)

# print(embeddings.shape)  # (num_samples, 440)

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

emb_2d = reducer.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    emb_2d[:, 0],
    emb_2d[:, 1],
    c=labels,
    cmap="tab10",
    s=8,
    alpha=0.7
)

plt.legend(*scatter.legend_elements(), title="Activity")
plt.title("JEPA Latent Space (Test Set)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.tight_layout()

visual_fn = f"jepa_latent_space_{exp_num}.png"
save_path = os.path.join(visualization_dir, visual_fn)
plt.savefig(save_path, dpi=300, bbox_inches="tight")
# plt.show()
