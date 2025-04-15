import numpy as np
import tensorflow as tf
import os
from utils.load_files import load_audio_file_paths, load_img_file_paths
from utils.feature_extraction import get_mel_spectrogram_tf, get_vector_features_tf, get_mel_sequence_tf
from utils.predict import predict_call_with_segments_plain, predict_call_with_segments_cnn_bilistm, predict_call_with_segments_multimodal, predict_call_with_segments_cnn_bigru, predict_call_with_segments_by_mean, predict_call_with_segments_hybrid

from audio_models import TransformerEncoderBlock

# ✅ 환경 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
print("🖥️ Physical GPUs:", tf.config.list_physical_devices('GPU'))

# ✅ 저장된 모델 경로
# model_path = "model/multi_cnn_bilstm_epoch_10_batch_32_final_2025-03-27_07-06-42.keras"
# IMAGE_WIDTH 400 MAX_LEN 400
# model_path = "model/multi_cnn_bilstm_epoch_10_batch_32_final_2025-03-27_08-36-17.keras"
# IMAGE_WIDTH 500 MAX_LEN 400
# model_path = "model/multi_cnn_bilstm_epoch_10_batch_32_final_2025-03-27_15-42-04.keras"
model_path = "model/multi_cnn_bigru_epoch_10_batch_32_final_2025-03-27_15-43-13.keras"

model = tf.keras.models.load_model(model_path, compile=False)
model.summary()
MAX_LEN = 400

def recover_audio_path_from_image(image_path, audio_root_dir):
    filename = os.path.basename(image_path).replace(".png", "")
    parts = filename.split("_")
    if len(parts) >= 3:
        core = "_".join(parts[1:-1])  # 앞뒤 제거
    else:
        core = parts[0]
    audio_filename = core + ".wav"
    return os.path.join(audio_root_dir, audio_filename)


# ✅ 테스트 파일 예측
audio_root_dir = "dataset/audio/tmp"
test_image_files = load_img_file_paths("dataset/img/tmp")

for file_path in test_image_files:
    # predict_call_with_segments_hybrid(recover_audio_path_from_image(file_path, audio_root_dir), file_path, model, MAX_LEN)
    predict_call_with_segments_by_mean(recover_audio_path_from_image(file_path, audio_root_dir), file_path, model, MAX_LEN)
    # predict_call_with_segments_multimodal(recover_audio_path_from_image(file_path, audio_root_dir), file_path, model, MAX_LEN)
