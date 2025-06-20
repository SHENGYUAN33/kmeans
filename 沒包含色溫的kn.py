import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from PIL import Image

# --------- 1. 資料集與Augmentation ---------
DATA_DIR = r"H:\dino\new_clusters"
BATCH_SIZE = 64
IMAGE_SIZE = 224

transform_simclr = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# SimCLR需一次產生兩個augmented views


class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.samples = [os.path.join(root, x) for x in os.listdir(root)
                        if x.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        # img = Image.open(img_path).convert("RGB")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 可以選擇 return None, None 然後在主程式裡過濾

        xi = self.transform(img)
        xj = self.transform(img)
        return xi, xj


# --------- 2. SimCLR Model定義 ---------


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimCLRNet(nn.Module):
    def __init__(self, base_model='resnet18', out_dim=128):
        super().__init__()
        self.encoder = getattr(models, base_model)(pretrained=False)
        feat_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.proj_head = ProjectionHead(feat_dim, out_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.proj_head(h)
        return h, z


device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimCLRNet(base_model='resnet18', out_dim=128).to(device)

# --------- 3. NT-Xent 損失（對比學習） ---------


def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    sim_i_j = torch.diag(sim, N)
    sim_j_i = torch.diag(sim, -N)
    positives = torch.cat([sim_i_j, sim_j_i], dim=0)
    mask = (~torch.eye(2*N, 2*N, dtype=bool)).float().to(device)
    exp_sim = torch.exp(sim) * mask
    negatives = exp_sim.sum(dim=1)
    loss = -torch.log(torch.exp(positives) / negatives)
    return loss.mean()


if __name__ == "__main__":
    # 訓練流程全部寫在這裡

    dataset = SimCLRDataset(DATA_DIR, transform_simclr)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=4, drop_last=True)
    # --------- 4. SimCLR訓練 ---------
    EPOCHS = 20
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xi, xj in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            xi, xj = xi.to(device), xj.to(device)
            _, zi = model(xi)
            _, zj = model(xj)
            loss = nt_xent_loss(zi, zj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(loader):.4f}")

    # --------- 5. 抽取全資料特徵 ---------
    model.eval()
    features = []
    img_paths = dataset.samples
    with torch.no_grad():
        for idx in tqdm(range(0, len(img_paths), BATCH_SIZE), desc="Extracting features"):
            batch_paths = img_paths[idx:idx+BATCH_SIZE]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            batch = torch.stack([transform_simclr(img)
                                for img in imgs]).to(device)
            h, _ = model(batch)
            features.append(h.cpu().numpy())
    features = np.concatenate(features, axis=0)

    # --------- 6. K-means 聚類 ---------
    n_clusters = 10  # 你可以自己決定分幾群
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(features)

    # --------- 7. 分群結果儲存 ---------
    output_dir = r"H:\dino\a"
    os.makedirs(output_dir, exist_ok=True)
    for i in range(n_clusters):
        cluster_dir = os.path.join(output_dir, f"cluster_{i}")
        os.makedirs(cluster_dir, exist_ok=True)
        idxs = np.where(cluster_ids == i)[0]
        # 每群最多存20張
        for j in idxs[:20]:
            img = Image.open(img_paths[j])
            img.save(os.path.join(cluster_dir, os.path.basename(img_paths[j])))

    print("分群結束，結果儲存於 simclr_kmeans_clusters/")
