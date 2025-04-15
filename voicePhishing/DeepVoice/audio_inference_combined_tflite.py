import numpy as np
import tensorflow as tf
import os
from utils.load_files import load_audio_file_paths
from utils.feature_extraction import get_spectrogram, get_combined_features, get_advanced_features
from utils.predict import predict_call_with_segments, predict_with_tflite, decode_audio_librosa

# ✅ 환경 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
print("🖥️ Physical GPUs:", tf.config.list_physical_devices('GPU'))

# ✅ 저장된 모델 경로
model_path = "quant_model/VGG19_BiLSTM_ensemble_nocudnn_lstm_best_model_2025-03-26_07-07-53.tflite"

with open(model_path, 'rb') as f:
    model_content = f.read()
print(model_content)
interpreter = tf.lite.Interpreter(model_content=model_content)
interpreter.allocate_tensors()

# ✅ 모델 입력/출력 텐서 정보
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# ✅ 테스트 오디오 파일 예측
test_audio_files = load_audio_file_paths("dataset/audio/tmp")

for file_path in test_audio_files:
    waveform = decode_audio_librosa(file_path)
    feature = get_advanced_features(waveform)
    feature = tf.expand_dims(feature, 0)  # (1, H, W, 1)

    prediction = predict_with_tflite(interpreter, feature)

    print(f"📞 [{os.path.basename(file_path)}]")
    print(f"🔍 Prediction: {prediction}")
    print(f"➡️ Deep Voice Probability: {prediction[0][1]:.4f}")
    print()

interpreter.close()