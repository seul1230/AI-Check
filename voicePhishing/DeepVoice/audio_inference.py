import numpy as np
import librosa
import tensorflow as tf
import os
import glob
from utils.load import load_audio_file_paths

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
print("🖥️ Physical GPUs:", tf.config.list_physical_devices('GPU'))

# ✅ **저장된 모델 로드**
model_path = "model/CNN_BiLSTM_best_model_2025-03-24_05-31-37.keras"
model = tf.keras.models.load_model(model_path, compile=False)

# ✅ **오디오 특징 추출 함수 (학습 시와 동일하게)**
def extract_audio_features(file_path, feature_type='mel', max_len=500):
    y, sr = librosa.load(file_path, sr=16000)

    # 📌 **너무 짧은 오디오는 무시**
    if len(y) < 1600:  # 0.1초 미만이면 스킵
        print(f"⚠ Warning: File {file_path} is too short and skipped.")
        return None

    # 📌 **데이터 증강 (Inference에서는 제외 가능)**
    # y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.8, 1.2))  # 학습 시만 사용

    # 적절한 n_fft 설정
    n_fft = min(len(y), 2048) if len(y) >= 256 else 256

    if feature_type == 'mel':
        feature = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=n_fft, hop_length=512, fmax=8000)
        feature = librosa.power_to_db(feature, ref=np.max)

    # 📌 **패딩 처리**
    pad_width = max_len - feature.shape[1]
    if pad_width > 0:
        feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
    else:
        feature = feature[:, :max_len]

    return feature

# ✅ **Inference 실행 함수**
def predict_audio(file_path):
    feature = extract_audio_features(file_path)
    if feature is None:
        return None  # 짧은 파일 무시

    # 📌 **입력 차원 변환 (모델과 동일한 형태로)**
    feature = feature[np.newaxis, ..., np.newaxis]  # (1, 128, max_len, 1)

    # 📌 **모델 예측**
    prediction = model.predict(feature)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    label_map = {0: "✅ Real Voice", 1: "‼️ Deepfake Voice"}
    print(f"🎙 {file_path} → Predicted: {label_map[predicted_class]} (Confidence: {confidence:.2%})")

    return predicted_class, confidence

# ✅ **테스트할 오디오 파일 지정**
test_audio_files = load_audio_file_paths("dataset/audio/tmp")

# ✅ **모든 테스트 파일에 대해 예측 수행**
for file in test_audio_files:
    predict_audio(file)