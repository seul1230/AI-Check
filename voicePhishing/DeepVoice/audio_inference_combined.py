import numpy as np
import tensorflow as tf
import os
from utils.load_files import load_audio_file_paths, load_img_file_paths
from utils.feature_extraction import get_spectrogram, get_combined_features, get_advanced_features, get_advanced_145_features_tf, get_mel_spectrogram_tf, get_vector_features_tf, get_mel_sequence_tf
from utils.predict import predict_call_with_segments_plain, predict_call_with_segments_cnn_bilistm, predict_call_with_segments_multimodal, predict_call_with_segments_cnn_bigru

from audio_models import TransformerEncoderBlock

# ✅ 환경 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
print("🖥️ Physical GPUs:", tf.config.list_physical_devices('GPU'))

# ✅ 저장된 모델 경로
# model_path = "model/VGG19_BiLSTM_ensemble_best_model_2025-03-25_08-32-23.keras"
# model_path = "model/VGG19_BiLSTM_ensemble_best_model_2025-03-25_16-07-09.keras" # -- 거의 다 잡힘
# model_path = "model/VGG19_BiLSTM_ensemble_best_model_2025-03-26_01-51-00.keras" # -- 이것도 거의 다 잡힘 진짜임
# model_path = "model/VGG19_BiGRU_best_model_2025-03-26_12-15-51.keras"
# model_path = "model/VGG19_BiGRU_best_model_2025-03-27_00-04-07.keras" # -- 거의 다,, ,된 거 같은데,,,, 
model_path = "model/multi_cnn_bilstm_final_2025-03-27_03-42-31.keras"
# model_path = "model/CNN_transformer_best_model_2025-03-26_11-23-30.keras"
# model_path = "quant_model/VGG19_BiLSTM_ensemble_best_model_2025-03-25_16-07-09.tflite"

model = tf.keras.models.load_model(model_path, compile=False)

# model = tf.keras.models.load_model(
#     model_path,
#     compile=False,
#     custom_objects={"TransformerEncoderBlock": TransformerEncoderBlock}
# )



# ✅ 테스트 파일 예측
test_audio_files = load_audio_file_paths("dataset/audio/tmp")
# audio_root_dir = "dataset/audio/tmp"
# test_image_files = load_img_file_paths("dataset/img/tmp")

for file_path in test_image_files:
    # predict_audio(file_path)
    # predict_call_with_segments(file_path, model)
    # predict_call_with_segments_cnn_bilistm(file_path, model)
    predict_call_with_segments_plain(file_path, model)
    # predict_call_with_segments_cnn_bigru(file_path, model)

