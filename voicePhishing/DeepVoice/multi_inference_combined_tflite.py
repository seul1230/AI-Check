# import numpy as np
# import tensorflow as tf
# import os

# from utils.load_files import load_audio_file_paths, load_img_file_paths
# from utils.feature_extraction import (
#     get_vector_features_tf,
#     get_mel_sequence_tf,
# )
# from utils.predict import decode_audio_librosa, decode_image_tf, predict_multimodal_tflite, predict_multimodal_tflite_segmented #, decode_image_tf
# from tensorflow.keras.preprocessing import image as keras_image

# # ✅ 환경 설정
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# print("🖥️ Physical GPUs:", tf.config.list_physical_devices('GPU'))

# # ✅ 모델 경로 (.tflite)
# model_path = "quant_model/multi_cnn_bigru_epoch_10_batch_32_final_2025-03-27_15-43-13.tflite"
# interpreter = tf.lite.Interpreter(model_path=model_path)
# interpreter.allocate_tensors()

# # ✅ 입력 전처리 설정
# IMG_HEIGHT = 128
# IMG_WIDTH = 500
# SEQUENCE_LEN = 400
# N_MELS = 128

# def pad_sequence(seq, target_len=SEQUENCE_LEN):
#     cur_len = tf.shape(seq)[0]
#     if cur_len < target_len:
#         pad = tf.zeros([target_len - cur_len, seq.shape[1]])
#         seq = tf.concat([seq, pad], axis=0)
#     else:
#         seq = seq[:target_len, :]
#     seq.set_shape([SEQUENCE_LEN, N_MELS])
#     return seq

# def save_mel_sequence_to_npy(wav_path, out_path):
#     waveform = decode_audio_librosa(wav_path)
#     mel_seq = get_mel_sequence_tf(waveform)  # shape: [time, n_mels]
#     np.save(out_path, mel_seq.numpy())
#     print(f"✅ Mel-sequence saved to {out_path}.npy")

# # ✅ 테스트용 파일 목록
# image_dir = "dataset/img/tmp"
# audio_dir = "dataset/audio/tmp"

# test_image_files = load_img_file_paths(image_dir)

# # ✅ 파일명 기반 매핑 (image → audio)
# def recover_audio_path_from_image(image_path):
#     filename = os.path.basename(image_path).replace(".png", "")
#     parts = filename.split("_")
#     if len(parts) >= 3:
#         core = "_".join(parts[1:-1])
#     else:
#         core = parts[0]
#     return os.path.join(audio_dir, core + ".wav")


# # ✅ 예측 루프 (세그먼트 + 전체 예측 모두 포함)
# for image_path in test_image_files:
#     audio_path = recover_audio_path_from_image(image_path)

#     results = predict_multimodal_tflite_segmented(
#         audio_path, image_path, interpreter,
#         IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT
#     )
    
#     if results is None:
#         continue  # 세그먼트가 없어서 생략된 경우

# print("✅ 모든 테스트 파일 예측 완료!")
# print(interpreter.get_input_details())
# interpreter._interpreter = None  # 안전하게 해제

import numpy as np
import tensorflow as tf
import os

from utils.load_files import load_audio_file_paths, load_img_file_paths
from utils.feature_extraction import (
    get_vector_features_tf,
    get_mel_sequence_tf,
)
from utils.predict import decode_audio_librosa, decode_image_tf, predict_multimodal_tflite, predict_multimodal_tflite_segmented
from tensorflow.keras.preprocessing import image as keras_image

# ✅ 환경 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
print("🖥️ Physical GPUs:", tf.config.list_physical_devices('GPU'))

# ✅ 모델 경로 (.tflite)
model_path = "quant_model/multi_cnn_bigru_epoch_10_batch_32_final_2025-03-27_15-43-13.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# ✅ 입력 전처리 설정
IMG_HEIGHT = 128
IMG_WIDTH = 500
SEQUENCE_LEN = 400
N_MELS = 128

def pad_sequence(seq, target_len=SEQUENCE_LEN):
    cur_len = tf.shape(seq)[0]
    if cur_len < target_len:
        pad = tf.zeros([target_len - cur_len, seq.shape[1]])
        seq = tf.concat([seq, pad], axis=0)
    else:
        seq = seq[:target_len, :]
    seq.set_shape([SEQUENCE_LEN, N_MELS])
    return seq

def save_mel_sequence_to_npy(wav_path, out_path):
    waveform = decode_audio_librosa(wav_path)
    mel_seq = get_mel_sequence_tf(waveform)  # shape: [time, n_mels]
    np.save(out_path, mel_seq.numpy())
    print(f"✅ Mel-sequence saved to {out_path}.npy")
    

# ✅ 테스트용 파일 목록
image_dir = "dataset/img/tmp"
audio_dir = "dataset/audio/tmp"

test_image_files = load_img_file_paths(image_dir)

# ✅ 파일명 기반 매핑 (image → audio)
def recover_audio_path_from_image(image_path):
    filename = os.path.basename(image_path).replace(".png", "")
    parts = filename.split("_")
    if len(parts) >= 3:
        core = "_".join(parts[1:-1])
    else:
        core = parts[0]
    return os.path.join(audio_dir, core + ".wav")


# ✅ 예측 루프 (세그먼트 + 전체 예측 모두 포함)
for image_path in test_image_files:
    audio_path = recover_audio_path_from_image(image_path)

    results = predict_multimodal_tflite_segmented(
        audio_path, image_path, interpreter,
        IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT
    )

    # ✅ Mel Sequence 저장 (파일명 기준)
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    debug_dir = "debug"
    os.makedirs(debug_dir, exist_ok=True)
    save_path = os.path.join(debug_dir, f"{basename}_mel_seq_python.npy")
    save_mel_sequence_to_npy(audio_path, save_path)
    print()

    if results is None:
        continue  # 세그먼트가 없어서 생략된 경우

print("✅ 모든 테스트 파일 예측 완료!")
print(interpreter.get_input_details())
interpreter._interpreter = None  # 안전하게 해제
