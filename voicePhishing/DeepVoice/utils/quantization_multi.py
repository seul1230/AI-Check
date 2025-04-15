import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

# from predict import decode_image
from feature_extraction import get_mel_spectrogram_tf, get_vector_features_tf, get_mel_sequence_tf
from preprocess import filter_nan_audio
# from load_files import load_audio_file_paths, load_image_file_paths

# 환경 설정
os.environ["CUDA_DEVICE_ORDER"] = ""
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# 기본 설정
IMG_HEIGHT = 128
IMG_WIDTH = 500
VECTOR_DIM = 10
SEQUENCE_LEN = 400
N_MELS = 128

# 오디오 디코딩
def decode_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
    return tf.squeeze(audio, axis=-1)

def decode_image(image_path, width):
    image_binary = tf.io.read_file(image_path)
    image = tf.io.decode_png(image_binary, channels=1)
    image = tf.image.resize(image, [IMG_HEIGHT, width])
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # 확실하게 shape 보정 (항상 3D로)
    image.set_shape([IMG_HEIGHT, width, 1])

    return image

# 시퀀스 패딩
# def pad_sequence(seq, target_len=SEQUENCE_LEN):
#     cur_len = tf.shape(seq)[0]
#     if cur_len < target_len:
#         pad = tf.zeros([target_len - cur_len, seq.shape[1]])
#         return tf.concat([seq, pad], axis=0)
#     return seq[:target_len, :]

def pad_sequence(seq, target_len=SEQUENCE_LEN):
    cur_len = tf.shape(seq)[0]
    if cur_len < target_len:
        pad = tf.zeros([target_len - cur_len, seq.shape[1]])
        seq = tf.concat([seq, pad], axis=0)
    else:
        seq = seq[:target_len, :]
    seq.set_shape([SEQUENCE_LEN, 128])  # 🔥 명시적 shape 설정
    return seq


# ✅ 대표 데이터셋 구성 함수 (TFLite 양자화 시 사용)
def representative_dataset():
    CSV_PATH = "dataset/filtered_merged_paths.csv"
    df = pd.read_csv(CSV_PATH)

    paths = list(zip(df["audio_path"].tolist(), df["image_path"].tolist()))
    random.seed(42)
    random.shuffle(paths)
    sample_paths = paths[:100]

    for audio_path, image_path in sample_paths:
        waveform = decode_audio(audio_path)
        vector = get_vector_features_tf(waveform)
        sequence = get_mel_sequence_tf(waveform)
        sequence = pad_sequence(sequence, target_len=SEQUENCE_LEN)
        sequence.set_shape([SEQUENCE_LEN, 128])  # ✅ 명시

        image = decode_image(image_path, IMG_WIDTH)
        image.set_shape([IMG_HEIGHT, IMG_WIDTH, 1])  # ✅ 명시

        # 확실하게 배치 차원까지 고정
        img_input = tf.expand_dims(image, 0)
        vec_input = tf.expand_dims(vector, 0)
        seq_input = tf.expand_dims(sequence, 0)

        img_input.set_shape([1, IMG_HEIGHT, IMG_WIDTH, 1])   # ✅
        vec_input.set_shape([1, 10])                         # ✅
        seq_input.set_shape([1, SEQUENCE_LEN, 128])          # ✅

        print("image:", img_input.shape)
        print("vector:", vec_input.shape)
        print("sequence:", seq_input.shape)

        yield [img_input, vec_input, seq_input]

# def representative_dataset():
#     CSV_PATH = "dataset/filtered_merged_paths.csv"
#     df = pd.read_csv(CSV_PATH)

#     # 무작위 샘플 선택 (예: 100개)
#     paths = list(zip(df["audio_path"].tolist(), df["image_path"].tolist()))
#     random.seed(42)
#     random.shuffle(paths)
#     sample_paths = paths[:100]
#     for audio_path, image_path in sample_paths:
#         waveform = decode_audio(audio_path)
#         vector = get_vector_features_tf(waveform)
#         sequence = get_mel_sequence_tf(waveform)
#         sequence = pad_sequence(sequence, target_len=SEQUENCE_LEN)
    
#         image = decode_image(image_path, IMG_WIDTH)
    
#         img_input = tf.expand_dims(image, 0)
#         vec_input = tf.expand_dims(vector, 0)
#         seq_input = tf.expand_dims(sequence, 0)
    
#         print("image:", img_input.shape)
#         print("vector:", vec_input.shape)
#         print("sequence:", seq_input.shape)
    
#         yield [img_input, vec_input, seq_input]

#     # for audio_path, image_path in sample_paths:
#     #     waveform = decode_audio(audio_path)
#     #     vector = get_vector_features_tf(waveform)
#     #     sequence = get_mel_sequence_tf(waveform)
#     #     sequence = pad_sequence(sequence, target_len=SEQUENCE_LEN)

#     #     image = decode_image(image_path, IMG_WIDTH)
#     #     # 배치 차원 추가 (각 input shape가 모델과 일치해야 함)
#     #     yield [
#     #         tf.expand_dims(image, 0),      # (1, 128, 500, 1)
#     #         tf.expand_dims(vector, 0),     # (1, 10)
#     #         tf.expand_dims(sequence, 0)    # (1, 400, 128)
#     #     ]

# ✅ 모델 로드 및 TFLite 변환
model_name = "multi_cnn_bigru_epoch_10_batch_32_final_2025-03-27_15-43-13"
model_path = f"model/{model_name}.keras"
model = tf.keras.models.load_model(model_path, compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = []
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,   # 기본 TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS        # Flex ops 사용 허용
]
converter._experimental_lower_tensor_list_ops = False
converter.experimental_enable_resource_variables = True

tflite_model = converter.convert()

# 저장
os.makedirs("quant_model", exist_ok=True)
with open(f"quant_model/{model_name}.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ 양자화 모델 저장 완료!")





converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
# 입력 shape을 -1로 지정하면 dynamic batch
converter.experimental_new_converter = True
tflite_model = converter.convert()