import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import argparse
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import umap
import joblib
from math import ceil

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str,
                    default=r"H:\\dino\\new_clusters", help='圖片資料夾路徑')
parser.add_argument('--output_dir', type=str,
                    default='cluster_outputs', help='輸出資料夾')
parser.add_argument('--max_k', type=int, default=10, help='自動選擇最佳分群上限')
args = parser.parse_args(args=[])

image_dir = args.image_dir
output_dir = args.output_dir
max_k = args.max_k
os.makedirs(output_dir, exist_ok=True)

# Device setting
print("🚀 CUDA 是否可用:", torch.cuda.is_available())
print("🖥️ 使用的 GPU:", torch.cuda.get_device_name(
    0) if torch.cuda.is_available() else "None")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet18 model
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(device)

# Image transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Feature extraction


def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(input_tensor).cpu().numpy().flatten()
    features = {
        'mean_R': np.mean(img_rgb[:, :, 0]),
        'mean_G': np.mean(img_rgb[:, :, 1]),
        'mean_B': np.mean(img_rgb[:, :, 2]),
        'std_R': np.std(img_rgb[:, :, 0]),
        'std_G': np.std(img_rgb[:, :, 1]),
        'std_B': np.std(img_rgb[:, :, 2]),
        'mean_H': np.mean(img_hsv[:, :, 0]),
        'mean_S': np.mean(img_hsv[:, :, 1]),
        'mean_V': np.mean(img_hsv[:, :, 2]),
        'mean_gray': np.mean(img_gray),
        'std_gray': np.std(img_gray),
        'blur_score': cv2.Laplacian(img_gray, cv2.CV_64F).var(),
        'color_temp': np.mean(img_rgb[:, :, 0]) - np.mean(img_rgb[:, :, 2])
    }
    for i, val in enumerate(embedding):
        features[f'emb_{i}'] = val
    return features


# Collect features
features_list = []
file_paths = []
for file in tqdm(os.listdir(image_dir), desc="擷取影像特徵"):
    if file.lower().endswith('.bmp'):
        full_path = os.path.join(image_dir, file)
        features = extract_features(full_path)
        if features:
            features_list.append(features)
            file_paths.append(full_path)

df_feat = pd.DataFrame(features_list)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_feat)

# Find best k using silhouette score


def find_best_k_silhouette(X_scaled, k_min=2, k_max=10):
    best_k = k_min
    best_score = -1
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
        score = silhouette_score(X_scaled, kmeans.labels_)
        print(f"k={k} → Silhouette Score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


best_k = find_best_k_silhouette(X_scaled, k_min=2, k_max=max_k)
print(f"✅ 使用最佳群數：{best_k}")

# KMeans clustering
kmeans = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df_feat['filepath'] = file_paths
df_feat['cluster'] = labels
joblib.dump(kmeans, os.path.join(output_dir, "kmeans_model.pkl"))

# UMAP visualization
reducer = umap.UMAP(n_components=2, random_state=42)
umap_result = reducer.fit_transform(X_scaled)
df_feat['umap_x'] = umap_result[:, 0]
df_feat['umap_y'] = umap_result[:, 1]

plt.figure(figsize=(10, 6))
for c in range(best_k):
    subset = df_feat[df_feat['cluster'] == c]
    plt.scatter(subset['umap_x'], subset['umap_y'],
                label=f'Cluster {c}', alpha=0.6)
plt.legend()
plt.title("UMAP Clustering Visualization")
plt.savefig(os.path.join(output_dir, "umap_clusters.png"))
plt.close()

# Save thumbnails


def plot_cluster_thumbnails(df, cluster_num, output_dir="cluster_thumbnails", images_per_row=6, max_images=100):
    os.makedirs(output_dir, exist_ok=True)
    files = df[df['cluster'] == cluster_num]['filepath'].tolist()
    count = min(len(files), max_images)
    rows = ceil(count / images_per_row)
    fig, axes = plt.subplots(rows, images_per_row, figsize=(
        images_per_row * 2.5, rows * 2.5))
    axes = axes.flatten()
    for i in range(len(axes)):
        if i < count:
            img = cv2.imread(files[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
            axes[i].axis('off')
        else:
            axes[i].remove()
    fig.suptitle(f'Cluster {cluster_num}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(output_dir, f'cluster_{cluster_num}.png')
    plt.savefig(out_path, dpi=100)
    plt.close()


for cluster_id in tqdm(sorted(df_feat['cluster'].unique()), desc="儲存各群縮圖"):
    plot_cluster_thumbnails(df_feat, cluster_id)

csv_path = os.path.join(output_dir, "cluster_results_with_all_features.csv")
df_feat.to_csv(csv_path, index=False)
print(f"\n✅ 聚類完成，結果已儲存至 {csv_path} 和 {output_dir} 資料夾")
