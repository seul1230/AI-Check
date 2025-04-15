import tensorflow as tf
import os

# ✅ GPU 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
print("🖥️ Physical GPUs:", tf.config.list_physical_devices('GPU'))

# ✅ 모델 파일명 (너가 저장한 .keras 경로)
# model_name = "mlp-256-128-v4-auc_epoch_50_lr_0.001_batch_32_none_final_2025-04-08_07-13-59"
# model_name = "mlp-256-128-64-v6-auc-2-1_epoch_50_lr_0.001_batch_32_none_final_2025-04-09_00-17-44"
model_name = "dacon-mlp-256-128-64-v7-auc-2-1_epoch_50_lr_0.001_batch_32_none_final_2025-04-09_07-38-01"
model_path = f"model/{model_name}.keras"
tflite_output_path = f"converted_model/{model_name}.tflite"

# ✅ 모델 불러오기
model = tf.keras.models.load_model(model_path)

# ✅ TFLite Converter 사용
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# ✅ 저장
with open(tflite_output_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite 모델 저장 완료: {tflite_output_path}")
