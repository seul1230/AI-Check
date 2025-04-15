import pandas as pd
import os
from load_files import load_audio_file_paths, load_img_file_paths
from preprocess import filter_nan_audio

# ✅ 경로 설정 (절대경로 기준 접근)
real_audio_dir = os.path.abspath("dataset/audio/train/real")
deep_audio_dir = os.path.abspath("dataset/audio/train/deepvoice")
real_image_dir = os.path.abspath("dataset/img/train/real")
deep_image_dir = os.path.abspath("dataset/img/train/deepvoice")

save_audio_csv = "dataset/filtered_audio_paths.csv"
save_image_csv = "dataset/filtered_image_paths.csv"
save_merged_csv = "dataset/filtered_merged_paths.csv"

# ✅ 오디오 파일 로드 및 NaN 필터링
real_audio_paths = load_audio_file_paths(real_audio_dir)
deep_audio_paths = load_audio_file_paths(deep_audio_dir)
clean_real_audio, clean_deep_audio, _, _ = filter_nan_audio(real_audio_paths, deep_audio_paths)

# ✅ 오디오 데이터프레임 구성
audio_data = []
for path in clean_real_audio:
    audio_data.append({"audio_path": path, "label": 0})
for path in clean_deep_audio:
    audio_data.append({"audio_path": path, "label": 1})
audio_df = pd.DataFrame(audio_data)
audio_df.to_csv(save_audio_csv, index=False)
print(f"✅ 오디오 경로 저장 완료: {save_audio_csv}")

# ✅ 이미지 경로 수집
real_image_paths = load_img_file_paths(real_image_dir)  # 이미지도 wav처럼 로드
deep_image_paths = load_img_file_paths(deep_image_dir)
image_data = []
for path in real_image_paths:
    image_data.append({"image_path": path, "label": 0})
for path in deep_image_paths:
    image_data.append({"image_path": path, "label": 1})
image_df = pd.DataFrame(image_data)
image_df.to_csv(save_image_csv, index=False)
print(f"✅ 이미지 경로 저장 완료: {save_image_csv}")

# ✅ 이미지 → 오디오 매핑 함수
def match_audio_path_from_image(image_path, audio_base_dir):
    filename = os.path.basename(image_path).replace(".png", "")
    parts = filename.split("_")
    if len(parts) >= 3:
        core = "_".join(parts[1:-1])  # 앞뒤 제거
    else:
        core = parts[0]
    audio_filename = core + ".wav"
    return os.path.relpath(os.path.join(audio_base_dir, audio_filename))

# ✅ 오디오 경로 복원 후 필터링
recovered_audio_paths = []
for row in image_df.itertuples():
    base_dir = real_audio_dir if row.label == 0 else deep_audio_dir
    audio_path = match_audio_path_from_image(row.image_path, base_dir)
    abs_audio_path = os.path.abspath(audio_path)
    if os.path.exists(abs_audio_path) and os.path.exists(row.image_path):
        recovered_audio_paths.append(audio_path)
    else:
        recovered_audio_paths.append(None)

image_df["audio_path"] = recovered_audio_paths
image_df = image_df.dropna(subset=["audio_path"])

# ✅ 상대경로로 변환
image_df["image_path"] = image_df["image_path"].apply(lambda x: os.path.relpath(x))
image_df["audio_path"] = image_df["audio_path"].apply(lambda x: os.path.relpath(x))

# ✅ 최종 저장
final_df = image_df[["audio_path", "image_path", "label"]]
final_df.to_csv(save_merged_csv, index=False)
print(f"✅ 병합된 오디오/이미지 경로 저장 완료: {save_merged_csv}")
print(f"👉 최종 샘플 수: {len(final_df)}")
