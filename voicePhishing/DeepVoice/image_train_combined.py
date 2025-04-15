# ✅ 멀티모달 모델 학습 코드 (CNN 단독 vs BiLSTM 앙상블 분기)
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import datetime
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy

from audio_models import build_multifeature_cnn_model, build_multifeature_cnn_bilstm_ensemble, build_multifeature_cnn_bilstm_concat_ensemble, build_multifeature_cnn_bigru_ensemble
from utils.load_files import load_audio_file_paths, load_img_file_paths
from utils.feature_extraction import get_mel_spectrogram_tf, get_vector_features_tf, get_mel_sequence_tf
from utils.preprocess import filter_nan_audio

# ✅ 실행 설정
model_name = "multi_cnn_bilstm"  # "cnn" or "cnn_bilstm"
# model_name = "multi_cnn_bigru"
# model_name = "multi_cnn_bilstm_concat"

IMG_HEIGHT = 128
IMG_WIDTH = 500
MAX_LEN = 400  # max time length

# ✅ 기본 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ✅ 저장 디렉토리 생성
today = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs("logs", exist_ok=True)
os.makedirs("ckpt", exist_ok=True)
os.makedirs("model", exist_ok=True)
log_dir = f"logs/{today}"


# ✅ 전처리 함수 (CNN 단독 / BiLSTM 앙상블)
def decode_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
    return tf.cast(tf.squeeze(audio, axis=-1), tf.float32)



def decode_image(image_path):
    image_binary = tf.io.read_file(image_path)
    image = tf.io.decode_png(image_binary, channels=1)  # 흑백
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])  # 리사이즈 추가!
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0, 1] 정규화

    return image


def preprocess_cnn(audio_path, image_path, label):
    waveform = decode_audio(audio_path)
    image = decode_image(image_path)
    vector = get_vector_features_tf(waveform)
    label = tf.one_hot(label, depth=2)
    return (image, vector), label


def pad_sequence(tensor, max_len=MAX_LEN):
    tensor = tensor[:max_len]
    pad_len = max_len - tf.shape(tensor)[0]
    return tf.pad(tensor, [[0, pad_len], [0, 0]])

def preprocess_bilstm(audio_path, image_path, label):
    waveform = decode_audio(audio_path)
    image = decode_image(image_path)
    vector = get_vector_features_tf(waveform)
    sequence = get_mel_sequence_tf(waveform)
    sequence = pad_sequence(sequence)  # <- 여기 패딩 추가
    label = tf.one_hot(label, depth=2)
    return (image, vector, sequence), label


CSV_PATH = "dataset/filtered_merged_paths.csv"
df = pd.read_csv(CSV_PATH)
X_audio = df["audio_path"].values
X_image = df["image_path"].values
y_labels = df["label"].values

# ✅ 데이터 분할
X_audio_train, X_audio_temp, X_image_train, X_image_temp, y_train, y_temp = train_test_split(
    X_audio, X_image, y_labels, test_size=0.3, stratify=y_labels, random_state=SEED
)
X_audio_val, X_audio_test, X_image_val, X_image_test, y_val, y_test = train_test_split(
    X_audio_temp, X_image_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
)

# ✅ Dataset 구성
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32

def create_dataset(audio_list, image_list, label_list, use_bilstm=False):
    ds = tf.data.Dataset.from_tensor_slices((audio_list, image_list, label_list))
    if use_bilstm:
        ds = ds.map(preprocess_bilstm, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(preprocess_cnn, num_parallel_calls=AUTOTUNE)
    return ds.shuffle(1000).batch(batch_size).prefetch(AUTOTUNE)


use_bilstm = model_name != "multi_cnn"
train_ds = create_dataset(X_audio_train, X_image_train, y_train, use_bilstm)
val_ds   = create_dataset(X_audio_val,   X_image_val,   y_val,   use_bilstm)
test_ds  = create_dataset(X_audio_test,  X_image_test,  y_test,  use_bilstm)

# ✅ 모델 생성
if model_name == "multi_cnn_bilstm":
    model = build_multifeature_cnn_bilstm_ensemble()
elif model_name == "multi_cnn_bilstm_concat":
    model = build_multifeature_cnn_bilstm_concat_ensemble()
elif model_name == "multi_cnn_bigru":
    model = build_multifeature_cnn_bigru_ensemble()
else:
    model = build_multifeature_cnn_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=CategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.Recall(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

epoch = 10

# ✅ 콜백 설정
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"ckpt/{model_name}_epoch_{epoch}_batch_{batch_size}_best_{today}.keras",
    save_best_only=True, monitor="val_recall", mode="max", verbose=1
)
early_stopping = EarlyStopping(monitor="val_recall", patience=3, restore_best_weights=True)
tensorboard_cb = TensorBoard(log_dir=log_dir)
lr_scheduler = ReduceLROnPlateau(monitor="val_recall", factor=0.5, patience=3, min_lr=1e-6)

print(f"{model_name}_epoch_{epoch}_batch_{batch_size}", IMG_WIDTH)

# ✅ 학습
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epoch,
    callbacks=[checkpoint_cb, early_stopping, tensorboard_cb, lr_scheduler]
)

# ✅ 저장
final_model_path = f"model/{model_name}_epoch_{epoch}_batch_{batch_size}_final_{today}.keras"
model.save(final_model_path)
print(f"✅ Model saved as: {final_model_path}")

# # ✅ 평가
# results = model.evaluate(test_ds)
# print("\n🎯 최종 테스트 결과:")
# for name, val in zip(model.metrics_names, results):
#     print(f"  - {name:<10}: {val:.4f}")

# ✅ 최종 테스트 성능 평가
metrics_names = ['Loss', 'Accuracy', 'Recall', 'AUC']
results = model.evaluate(test_ds, verbose=1)
print(final_model_path)
print("🎯 Final Evaluation Results:")
for name, val in zip(metrics_names, results):
    print(f"  - {name:<10}: {val:.4f}")


# ✅ 예측
# print("\n📊 예측 결과 분석 중...")
# y_pred = []
# for audio_path, image_path in zip(X_audio_test, X_image_test):
#     waveform = decode_audio(audio_path)
#     image = decode_image(image_path)
#     vector = get_vector_features_tf(waveform)
#     image = tf.expand_dims(image, 0)
#     vector = tf.expand_dims(vector, 0)

#     if use_bilstm:
#         sequence = get_mel_sequence_tf(waveform)
#         sequence = tf.expand_dims(sequence, 0)
#         pred = model.predict([image, vector, sequence], verbose=0)
#     else:
#         pred = model.predict([image, vector], verbose=0)

#     y_pred.append(tf.argmax(pred, axis=1).numpy()[0])

# print(classification_report(y_test, y_pred, target_names=["Real", "Deep"]))

print("\n📊 예측 결과 분석 중...")
y_pred = []
y_true = []

for batch in test_ds:
    inputs, labels = batch
    preds = model.predict(inputs, verbose=0)
    y_pred.extend(tf.argmax(preds, axis=1).numpy())
    y_true.extend(tf.argmax(labels, axis=1).numpy())

print(classification_report(y_true, y_pred, target_names=["Real", "Deep"]))
