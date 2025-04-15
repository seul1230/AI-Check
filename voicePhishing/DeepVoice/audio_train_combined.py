import numpy as np
import tensorflow as tf
import os
import datetime
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.losses import CategoricalCrossentropy

from audio_models import build_cnn, build_cnn_bilstm, build_vgg19_bilstm, build_vgg19_bilstm_ensemble, build_vgg19_bilstm_ensemble_cpu_lstm, build_cnn_transformer, build_vgg19_bigru_ensemble
from utils.load_files import load_audio_file_paths
from utils.feature_extraction import get_spectrogram, get_combined_features, get_advanced_features, get_advanced_145_features, get_advanced_145_features_tf
from utils.preprocess import filter_nan_audio

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
print("🖥️ Physical GPUs:", tf.config.list_physical_devices('GPU'))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

today = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_dir = "ckpt"
final_model_dir = "model"
log_dir = f"logs/{today}"

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(final_model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# ✅ 데이터 로딩
real_voice_paths = load_audio_file_paths("dataset/audio/train/real")
deep_voice_paths = load_audio_file_paths("dataset/audio/train/deepvoice")

print(f"📊 Real Voice 개수: {len(real_voice_paths)}")
print(f"📊 Deep Voice 개수: {len(deep_voice_paths)}")

# min_samples = min(len(real_voice_paths), len(deep_voice_paths))
# real_voice_paths = real_voice_paths[:min_samples]
# deep_voice_paths = deep_voice_paths[:min_samples]

# X_paths = np.array(real_voice_paths + deep_voice_paths)
# y_labels = np.array([0] * len(real_voice_paths) + [1] * len(deep_voice_paths))

clean_real, clean_deep, removed_real, removed_deep = filter_nan_audio(real_voice_paths, deep_voice_paths)

X_paths = np.array(list(clean_real) + list(clean_deep))
y_labels = np.array([0] * len(clean_real) + [1] * len(clean_deep))


# ✅ 전체를 먼저 train(70%) / temp(30%) 로 나눈 뒤
X_train, X_temp, y_train, y_temp = train_test_split(
    X_paths, y_labels, test_size=0.30, random_state=SEED, stratify=y_labels, shuffle=True
)

# ✅ temp(30%) 를 다시 validation(15%) / test(15%) 로 나눔
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp, shuffle=True
)


def decode_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, sr = tf.audio.decode_wav(audio_binary, desired_channels=1)
    # return tf.squeeze(audio, axis=-1)
    return tf.cast(tf.squeeze(audio, axis=-1), tf.float32)

def preprocess(file_path, label):
    audio = decode_audio(file_path)
    spectrogram = get_spectrogram(audio)
    label = tf.one_hot(label, depth=2)
    return spectrogram, label

def preprocess_combined(file_path, label):
    audio = decode_audio(file_path)
    features = get_advanced_145_features_tf(audio)
    label = tf.one_hot(tf.cast(label, tf.int32), depth=2)  # One-hot encoding
    return features, label

# ✅ 모델 선택
model_name = "VGG19_BiGRU"  # <- 변경 가능

# ✅ TensorFlow Dataset 전처리
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
if model_name == "CNN_transformer":
    AUTOTUNE = 2
    batch_size = 8

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(1000).map(preprocess_combined, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.map(preprocess_combined, num_parallel_calls=AUTOTUNE)  # 또는 preprocess
val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.map(preprocess_combined, num_parallel_calls=AUTOTUNE)  # 또는 preprocess
test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)

print(f"train:val:test= {len(train_ds)} : {len(val_ds)} : {len(test_ds)}")

learning_rate = 0.0001
if model_name == "CNN_BiLSTM":
    epoch = 5
    model = build_cnn_bilstm(142, 0.5)
elif model_name == "VGG19_BiLSTM":
    epoch = 10
    model = build_vgg19_bilstm(299, 0.3)
elif model_name == "VGG19_BiLSTM_ensemble":
    epoch = 15
    model = build_vgg19_bilstm_ensemble(dropout_rate=0.5)
elif model_name == "VGG19_BiLSTM_ensemble_nocudnn_lstm":
    epoch = 10
    model = build_vgg19_bilstm_ensemble_cpu_lstm()
elif model_name == "CNN_transformer":
    epoch = 10
    model = build_cnn_transformer((145, 500, 1))
elif model_name == "VGG19_BiGRU":
    epoch = 15
    # learning_rate = 0.0001
    model = build_vgg19_bigru_ensemble((145, None, 1))
else:
    model = build_cnn()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=CategoricalCrossentropy(from_logits=False),#, label_smoothing=1e-5),
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

print("model name >> ", model_name)

# ✅ 학습률 로깅
def log_lr(epoch, logs):
    lr = model.optimizer.learning_rate.numpy()
    with tf.summary.create_file_writer(log_dir).as_default():
        tf.summary.scalar("learning_rate", data=lr, step=epoch)

lr_logger = LambdaCallback(on_epoch_end=log_lr)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"{checkpoint_dir}/{model_name}_best_model_{today}_batch_{batch_size}.keras",
    save_best_only=True, monitor="val_recall", mode="max", verbose=1
)

early_stopping = EarlyStopping(monitor="val_recall", patience=3, restore_best_weights=True)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
lr_scheduler = ReduceLROnPlateau(monitor="val_recall", factor=0.5, patience=3, min_lr=1e-6, verbose=1)


# ✅ 학습
history = model.fit(
    train_ds, validation_data=val_ds, epochs=epoch,
    callbacks=[checkpoint_callback, early_stopping, tensorboard_callback, lr_scheduler, lr_logger]
)


# ✅ 저장
final_model_path = f"{final_model_dir}/{model_name}_best_model_{today}.keras"
model.save(final_model_path, save_format="tf")

print(f"✅ Model saved as: {final_model_path}")
print(f"✅ TensorBoard logs saved at: {log_dir}")

# ✅ 최종 테스트 성능 평가
metrics_names = ['Loss', 'Accuracy', 'Precision', 'Recall', 'AUC']
results = model.evaluate(test_ds, verbose=1)
print("🎯 Final Evaluation Results:")
for name, val in zip(metrics_names, results):
    print(f"  - {name:<10}: {val:.4f}")


# ✅ 예측
y_pred = []
for file_path in X_test:
    waveform = decode_audio(file_path)
    feature = get_advanced_145_features_tf(waveform)
    feature = tf.expand_dims(feature, 0)  # (1, 142, time, 1)

    prediction = model.predict(feature, verbose=0)
    predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
    y_pred.append(predicted_class)

target_names = ["Real", "Deep"]
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))


