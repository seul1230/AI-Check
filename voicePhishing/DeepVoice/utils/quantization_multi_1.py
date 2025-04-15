import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from feature_extraction import get_mel_spectrogram_tf, get_vector_features_tf, get_mel_sequence_tf
from preprocess import filter_nan_audio
# from load_files import load_audio_file_paths, load_image_file_paths

# 환경 설정
os.environ["CUDA_DEVICE_ORDER"] = ""
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "9"

# 기본 설정
IMG_HEIGHT = 128
IMG_WIDTH = 500
VECTOR_DIM = 10
SEQUENCE_LEN = 400
N_MELS = 128

# 오디오 및 이미지 디코딩
def decode_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
    return tf.squeeze(audio, axis=-1)

def decode_image(image_path):
    image_binary = tf.io.read_file(image_path)
    image = tf.io.decode_png(image_binary, channels=1)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

# 시퀀스 패딩
def pad_sequence(seq, target_len=SEQUENCE_LEN):
    cur_len = tf.shape(seq)[0]
    if cur_len < target_len:
        pad = tf.zeros([target_len - cur_len, seq.shape[1]])
        return tf.concat([seq, pad], axis=0)
    return seq[:target_len, :]



# ✅ 대표 데이터셋 구성 함수
def representative_dataset():
    CSV_PATH = "dataset/filtered_merged_paths.csv"
    df = pd.read_csv(CSV_PATH)

    # 🔀 무작위 샘플 선택 (100개)
    paths = list(zip(df["audio_path"].tolist(), df["image_path"].tolist()))
    random.seed(42)
    random.shuffle(paths)
    sample_paths = paths[:100]

    for audio_path, image_path in sample_paths:
        waveform = decode_audio(audio_path)
        vector = get_vector_features_tf(waveform)
        sequence = get_mel_sequence_tf(waveform)
        sequence = pad_sequence(sequence, target_len=400)

        image = decode_image(image_path, img_width=IMG_WIDTH)

        # 배치 차원 추가
        yield [tf.expand_dims(image, 0), tf.expand_dims(vector, 0), tf.expand_dims(sequence, 0)]

# ✅ 모델 로드 및 TFLite 변환
# model_name = "multi_cnn_bigru_epoch_10_batch_32_final_2025-03-27_15-43-13"
model_name = "multi_cnn_bilstm_epoch_10_batch_32_final_2025-03-27_15-42-04"
model_path = f"model/{model_name}.keras"
model = tf.keras.models.load_model(model_path, compile=False)


# TensorFlow Lite Converter를 이용한 정수 양자화 진행 (수정된 부분)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# 대표 데이터셋 설정
converter.representative_dataset = representative_dataset

# Resource 변수 사용 활성화
converter.experimental_enable_resource_variables = True
# TensorList 관련 연산 내림(lowering) 기능 비활성화
converter._experimental_lower_tensor_list_ops = False

# Flex Ops(SELECT_TF_OPS)도 허용하여 변환 (정수 양자화와 함께 사용)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# (선택 사항) 입력과 출력도 int8로 변환 (모바일에서의 효율성을 위해)
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# 변환 수행
tflite_model_full_int = converter.convert()





# 저장
os.makedirs("quant_model", exist_ok=True)
with open(f"quant_model/{model_name}.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ 양자화 모델 저장 완료!")
