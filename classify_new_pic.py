import os
import cv2
import numpy as np
import joblib
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler

# 路徑
new_image_dir = r"H:\glaucoma\H409_1258P\PHO"  # 要分類的新影像資料夾
kmeans_path = r"H:\knnttt\aaa\cluster_outputs\kmeans_model.pkl"
output_folder = r"H:\knnttt\aaa\classified_images"

# 載入模型
kmeans = joblib.load(kmeans_path)

# 載入 ResNet 模型
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 特徵擷取（完全跟訓練階段一致）


def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 深度學習嵌入
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


# 開始處理每一張新圖
for filename in os.listdir(new_image_dir):
    if filename.lower().endswith(".bmp"):
        full_path = os.path.join(new_image_dir, filename)
        feats = extract_features(full_path)
        if feats:
            feats_arr = np.array(list(feats.values())).reshape(1, -1)
            cluster = kmeans.predict(feats_arr)[0]

            # 建立資料夾並搬移檔案
            cluster_dir = os.path.join(output_folder, f"cluster_{cluster}")
            os.makedirs(cluster_dir, exist_ok=True)
            dst_path = os.path.join(cluster_dir, filename)
            cv2.imwrite(dst_path, cv2.imread(full_path))
            print(f"✅ {filename} → Cluster {cluster}")
        else:
            print(f"⚠️ 無法讀取或處理：{filename}")
