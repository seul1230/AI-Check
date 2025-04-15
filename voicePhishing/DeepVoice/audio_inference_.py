import numpy as np
import tensorflow as tf
import os
from utils.load import load_audio_file_paths

# ✅ 환경 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
print("🖥️ Physical GPUs:", tf.config.list_physical_devices('GPU'))

# ✅ 저장된 모델 로드
model_path = "model/VGG19_BiLSTM_best_model_2025-03-24_08-36-46.keras"
model = tf.keras.models.load_model(model_path, compile=False)

# ✅ 오디오 디코딩 (학습 시와 동일하게)
def decode_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, sr = tf.audio.decode_wav(audio_binary, desired_channels=1)
    return tf.cast(tf.squeeze(audio, axis=-1), tf.float32)

# ✅ Mel Spectrogram 추출 함수 (학습 때와 동일)
def get_spectrogram(waveform, sr=16000, max_frames=500):
    stft = tf.signal.stft(waveform, frame_length=512, frame_step=256)
    spectrogram = tf.abs(stft)

    num_spectrogram_bins = spectrogram.shape[-1]
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=128,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sr,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0
    )
    mel_spectrogram = tf.matmul(spectrogram, mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    log_mel_spectrogram = log_mel_spectrogram[:max_frames, :]
    pad_len = tf.maximum(0, max_frames - tf.shape(log_mel_spectrogram)[0])
    log_mel_spectrogram = tf.pad(log_mel_spectrogram, [[0, pad_len], [0, 0]])

    log_mel_spectrogram = tf.transpose(log_mel_spectrogram)  # (128, time)
    log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, -1)  # (128, time, 1)
    return log_mel_spectrogram

# ✅ 예측 함수
def predict_audio(file_path):
    try:
        waveform = decode_audio(file_path)
        feature = get_spectrogram(waveform)
        feature = tf.expand_dims(feature, 0)  # (1, 128, 500, 1)

        prediction = model.predict(feature, verbose=0)
        predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
        confidence = tf.reduce_max(prediction).numpy()

        label_map = {0: "✅ Real Voice", 1: "‼️ Deepfake Voice"}
        print(f"🎙 {file_path.split('/')[-1]} → Predicted: {label_map[predicted_class]} (Confidence: {confidence:.2%})")

        return predicted_class, confidence
    except Exception as e:
        print(f"⚠ Error processing {file_path}: {e}")
        return None

# ✅ 테스트 오디오 파일 예측
test_audio_files = load_audio_file_paths("dataset/audio/tmp")

for file_path in test_audio_files:
    predict_audio(file_path)
