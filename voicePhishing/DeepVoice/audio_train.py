import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf
import os
import glob
import datetime
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, LambdaCallback

# 모델 불러오기
from audio_models import build_cnn, build_cnn_bilstm
from utils.load import load_audio_file_paths

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
print("🖥️ Physical GPUs:", tf.config.list_physical_devices('GPU'))


# # GPU 설정
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError as e:
#         print(e)

# 랜덤 시드 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 체크포인트 및 TensorBoard 설정
today = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_dir = "ckpt"
final_model_dir = "model"
log_dir = f"logs/{today}"

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(final_model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# ✅ 데이터 전처리 함수
def process_audio(file_path, max_len=500):
    try:
        y, sr = sf.read(file_path, samplerate=16000)
    except:
        y, sr = librosa.load(file_path, sr=16000)

    if len(y) < 1600:
        return None

    n_fft = min(len(y), 2048) if len(y) >= 256 else 256
    feature = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=n_fft, hop_length=512, fmax=8000)
    feature = librosa.power_to_db(feature, ref=np.max)

    pad_width = max_len - feature.shape[1]
    feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant') if pad_width > 0 else feature[:, :max_len]

    return feature

def data_generator(file_paths, labels, batch_size=32):
    while True:
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            batch_features = [process_audio(file) for file in batch_files if process_audio(file) is not None]

            if batch_features:
                batch_features = np.array(batch_features)[..., np.newaxis]
                batch_labels = to_categorical(batch_labels, num_classes=2)
                yield batch_features, batch_labels

# ✅ 데이터 로딩
# real_voice_paths = glob.glob("/Users/seul/Downloads/original_kids_train/*.wav")
# deep_voice_paths = glob.glob("/Users/seul/seulhub/ssafy_12/pjt/AIcheck/YourTTS/outputs_v2_generated_kids/*.wav")

real_voice_paths = load_audio_file_paths("dataset/audio/train/real")
deep_voice_paths = load_audio_file_paths("dataset/audio/train/deepvoice")

min_samples = min(len(real_voice_paths), len(deep_voice_paths))
real_voice_paths = real_voice_paths[:min_samples]
deep_voice_paths = deep_voice_paths[:min_samples]

X_paths = np.array(real_voice_paths + deep_voice_paths)
y_labels = np.array([0] * len(real_voice_paths) + [1] * len(deep_voice_paths))

X_train, X_test, y_train, y_test = train_test_split(
    X_paths, y_labels, test_size=0.2, random_state=SEED, stratify=y_labels, shuffle=True
)

batch_size = 32
train_data_gen = data_generator(X_train, y_train, batch_size)
test_data_gen = data_generator(X_test, y_test, batch_size)

# ✅ 모델 선택
model_name = "CNN_BiLSTM"  # "CNN" 또는 "CNN_BiLSTM" 선택 가능
model = build_cnn_bilstm() if model_name == "CNN_BiLSTM" else build_cnn()

# ✅ TensorBoard 학습률 로깅 콜백 추가
def log_lr(epoch, logs):
    lr = model.optimizer.learning_rate.numpy()
    with tf.summary.create_file_writer(log_dir).as_default():
        tf.summary.scalar("learning_rate", data=lr, step=epoch)

lr_logger = LambdaCallback(on_epoch_end=log_lr)

# ✅ 콜백 설정 (TensorBoard + Learning Rate Logging 포함)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"{checkpoint_dir}/{model_name}_best_model_{batch_size}_{today}.keras",
    save_best_only=True, monitor="val_loss", mode="min", verbose=1
)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

# ✅ Learning Rate Scheduler 추가
lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3, 
    min_lr=1e-6,
    verbose=1
)

# ✅ 모델 학습
steps_per_epoch = len(X_train) // batch_size
validation_steps = len(X_test) // batch_size

history = model.fit(
    train_data_gen, validation_data=test_data_gen, epochs=10,
    steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
    callbacks=[checkpoint_callback, early_stopping, tensorboard_callback, lr_scheduler, lr_logger]
)

# ✅ 모델 저장
final_model_path = f"{final_model_dir}/{model_name}_best_model_{today}_batch_{batch_size}.keras"
model.save(final_model_path)

print(f"✅ Model saved as: {final_model_path}")
print(f"✅ TensorBoard logs saved at: {log_dir}")
