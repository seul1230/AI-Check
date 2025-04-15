import os
import numpy as np
import random
import tensorflow as tf

from feature_extraction import get_advanced_features, get_advanced_145_features, get_advanced_145_features_tf
from preprocess import filter_nan_audio
from load_files import load_audio_file_paths

# # ✅ 환경 설정
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
print("🖥️ Physical GPUs:", tf.config.list_physical_devices('GPU'))

def decode_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, sr = tf.audio.decode_wav(audio_binary, desired_channels=1)
    return tf.cast(tf.squeeze(audio, axis=-1), tf.float32)

def representative_dataset():
    # 오디오 경로 불러오기
    real_voice_paths = load_audio_file_paths("dataset/audio/train/real")
    deep_voice_paths = load_audio_file_paths("dataset/audio/train/deepvoice")

    # NaN 제거
    clean_real, clean_deep, _, _ = filter_nan_audio(real_voice_paths, deep_voice_paths)
    all_paths = list(clean_real) + list(clean_deep)

    # 랜덤 섞기 + 샘플 100개 선택
    random.seed(42)
    random.shuffle(all_paths)
    sample_paths = all_paths[:100]

    # 전처리 및 yield
    for file_path in sample_paths:
        waveform = decode_audio(file_path)
        features = get_advanced_145_features_tf(waveform)
        features = tf.expand_dims(features, axis=0)  # batch dim
        yield [features]

# model/VGG19_BiLSTM_ensemble_best_model_2025-03-26_01-51-00.keras
model_name = "VGG19_BiGRU_best_model_2025-03-27_00-04-07"
model = tf.keras.models.load_model(f"model/{model_name}.keras")

# # ✅ TFLite 변환기 생성
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_model = converter.convert()


converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 옵션 설정
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset  # 적용했다면
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False
converter.experimental_enable_resource_variables = True

tflite_model = converter.convert()

# 파일 저장
with open(f"quant_model/{model_name}.tflite", "wb") as f:
    f.write(tflite_model)
