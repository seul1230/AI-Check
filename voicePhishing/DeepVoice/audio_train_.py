import numpy as np
import tensorflow as tf
import os
import datetime
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, LambdaCallback
from audio_models import build_cnn, build_cnn_bilstm
from utils.load import load_audio_file_paths

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

# min_samples = min(len(real_voice_paths), len(deep_voice_paths))
# real_voice_paths = real_voice_paths[:min_samples]
# deep_voice_paths = deep_voice_paths[:min_samples]

X_paths = np.array(real_voice_paths + deep_voice_paths)
y_labels = np.array([0] * len(real_voice_paths) + [1] * len(deep_voice_paths))

X_train, X_test, y_train, y_test = train_test_split(
    X_paths, y_labels, test_size=0.25, random_state=SEED, stratify=y_labels, shuffle=True
)

# ✅ TensorFlow Dataset 전처리
AUTOTUNE = tf.data.AUTOTUNE

def decode_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, sr = tf.audio.decode_wav(audio_binary, desired_channels=1)
    # return tf.squeeze(audio, axis=-1)
    return tf.cast(tf.squeeze(audio, axis=-1), tf.float32)

def get_spectrogram(waveform, sr=16000, max_frames=500):
    # STFT
    stft = tf.signal.stft(waveform, frame_length=512, frame_step=256)
    spectrogram = tf.abs(stft)

    # Mel filter bank
    num_spectrogram_bins = spectrogram.shape[-1]
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=128,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sr,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0
    )
    mel_spectrogram = tf.matmul(spectrogram, mel_weight_matrix)

    # Log scale
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # Padding or trimming
    log_mel_spectrogram = log_mel_spectrogram[:max_frames, :]
    pad_len = tf.maximum(0, max_frames - tf.shape(log_mel_spectrogram)[0])
    log_mel_spectrogram = tf.pad(log_mel_spectrogram, [[0, pad_len], [0, 0]])

    # Transpose & Add channel dim
    log_mel_spectrogram = tf.transpose(log_mel_spectrogram)  # (128, time)
    log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, -1)
    return log_mel_spectrogram


def preprocess(file_path, label):
    audio = decode_audio(file_path)
    spectrogram = get_spectrogram(audio)
    label = tf.one_hot(label, depth=2)
    return spectrogram, label

batch_size = 32

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(1000).map(preprocess, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)

# ✅ 모델 선택
model_name = "CNN_BiLSTM"
model = build_cnn_bilstm() if model_name == "CNN_BiLSTM" else build_cnn()

# ✅ 학습률 로깅
def log_lr(epoch, logs):
    lr = model.optimizer.learning_rate.numpy()
    with tf.summary.create_file_writer(log_dir).as_default():
        tf.summary.scalar("learning_rate", data=lr, step=epoch)

lr_logger = LambdaCallback(on_epoch_end=log_lr)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"{checkpoint_dir}/{model_name}_best_model_{today}_batch_{batch_size}.keras",
    save_best_only=True, monitor="val_loss", mode="min", verbose=1
)
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# ✅ 학습
history = model.fit(
    train_ds, validation_data=test_ds, epochs=5,
    callbacks=[checkpoint_callback, early_stopping, tensorboard_callback, lr_scheduler, lr_logger]
)

# ✅ 저장
final_model_path = f"{final_model_dir}/{model_name}_best_model_{today}.keras"
model.save(final_model_path)

# ✅ 최종 테스트 성능 평가
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"🎯 Final Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")

print(f"✅ Model saved as: {final_model_path}")
print(f"✅ TensorBoard logs saved at: {log_dir}")
