import os
import pandas as pd

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# print("🖥️ Physical GPUs:", tf.config.list_physical_devices('GPU'))

# 데이터 불러오기
urlset_df = pd.read_csv("dataset/urlset_.csv")       # PhishStorm
urlhaus_df = pd.read_csv("dataset/urlhaus.csv")      # URLhaus

# 1. PhishStorm: 컬럼 이름 변경 및 정제
urlset_df = urlset_df.rename(columns={"domain": "url"})
urlset_df["url"] = urlset_df["url"].str.lower().str.strip().str.rstrip("/")
urlset_df["label"] = pd.to_numeric(urlset_df["label"], errors="coerce").fillna(0).astype(int)
urlset_df = urlset_df[["url", "label"]]
urlset_df["label"] = urlset_df["label"].astype(int)  # 0: 정상, 1: 피싱

# 2. URLhaus: malware만 있으므로 label = 1
urlhaus_df["url"] = urlhaus_df["url"].str.lower().str.strip().str.rstrip("/")
urlhaus_df["label"] = 1
urlhaus_df = urlhaus_df[["url", "label"]]

# 3. 병합 후 중복 제거
combined_df = pd.concat([urlset_df, urlhaus_df], ignore_index=True)
combined_df = combined_df.drop_duplicates(subset="url")

# 결과 확인
print("총 데이터 수:", len(combined_df))
print(combined_df["label"].value_counts())

# CSV로 저장
combined_df.to_csv("dataset/final_malicious_url_dataset.csv", index=False)
