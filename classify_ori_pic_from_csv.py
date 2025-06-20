import pandas as pd
import os
import shutil

# ✅ 讀取分群結果（假設你有一個存好的 CSV）
# 要包含 filepath 和 cluster 欄位
df = pd.read_csv(
    r"H:\knnttt\aaa\cluster_outputs\cluster_results_with_all_features.csv")


def save_cluster_images(df, output_base_dir="clustered_output"):
    os.makedirs(output_base_dir, exist_ok=True)

    for cluster_num in df['cluster'].unique():
        cluster_dir = os.path.join(output_base_dir, f"cluster_{cluster_num}")
        os.makedirs(cluster_dir, exist_ok=True)

        cluster_files = df[df['cluster'] == cluster_num]['filepath'].tolist()
        for file in cluster_files:
            try:
                shutil.copy(file, os.path.join(
                    cluster_dir, os.path.basename(file)))
            except Exception as e:
                print(f"❌ 無法複製 {file}: {e}")


print("📂 分類影像儲存中...")
save_cluster_images(df, output_base_dir="clustered_output")
print("✅ 完成分類影像資料夾建立")
