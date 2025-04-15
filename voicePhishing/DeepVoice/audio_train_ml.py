from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from utils.load_files import load_audio_file_paths
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
print("🖥️ Physical GPUs:", tf.config.list_physical_devices('GPU'))

def get_combined_features(waveform, sr=16000, max_frames=500):
    stft = tf.signal.stft(waveform, frame_length=512, frame_step=256)
    spectrogram = tf.abs(stft)

    num_spectrogram_bins = spectrogram.shape[-1]
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=128,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sr,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0
    )
    mel_spectrogram = tf.matmul(spectrogram, mel_weight_matrix)
    log_mel = tf.math.log(mel_spectrogram + 1e-6)

    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :13]

    centroid = tf.reduce_sum(
        tf.cast(tf.range(num_spectrogram_bins), tf.float32)[tf.newaxis, :] * spectrogram,
        axis=-1
    ) / (tf.reduce_sum(spectrogram, axis=-1) + 1e-6)
    pitch = tf.expand_dims(centroid, -1)

    def pad_or_trim(tensor):
        tensor = tensor[:max_frames, :]
        pad_len = tf.maximum(0, max_frames - tf.shape(tensor)[0])
        return tf.pad(tensor, [[0, pad_len], [0, 0]])

    mel = pad_or_trim(log_mel)     # (T, 128)
    mfcc = pad_or_trim(mfccs)      # (T, 13)
    pitch = pad_or_trim(pitch)     # (T, 1)

    combined = tf.concat([mel, mfcc, pitch], axis=-1)  # (T, 142)
    combined = tf.transpose(combined, [1, 0])          # (142, time)
    combined = tf.expand_dims(combined, -1)            # (142, time, 1)
    return combined


# ✅ 오디오 feature 추출
def extract_flat_features(file_paths):
    features = []
    for path in file_paths:
        waveform = decode_audio(path)
        feat = get_combined_features(waveform)  # (142, time, 1)
        feat = tf.squeeze(feat, axis=-1)        # (142, time)
        flat = tf.reshape(feat, [-1])            # Flatten: (142*time,)
        features.append(flat.numpy())
    return np.array(features)

# ✅ 오디오 디코딩
def decode_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, sr = tf.audio.decode_wav(audio_binary, desired_channels=1)
    return tf.cast(tf.squeeze(audio, axis=-1), tf.float32)

# ✅ TensorFlow Dataset 전처리
AUTOTUNE = tf.data.AUTOTUNE

# ✅ 데이터 로딩
real_paths = load_audio_file_paths("dataset/audio/train/real")
deep_paths = load_audio_file_paths("dataset/audio/train/deepvoice")

X = extract_flat_features(real_paths + deep_paths)
y = np.array([0] * len(real_paths) + [1] * len(deep_paths))

# ✅ 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_paths, y_labels, test_size=0.30, random_state=SEED, stratify=y_labels, shuffle=True
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp, shuffle=True
)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(1000).map(preprocess_combined, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.map(preprocess_combined, num_parallel_calls=AUTOTUNE)  # 또는 preprocess
val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.map(preprocess_combined, num_parallel_calls=AUTOTUNE)  # 또는 preprocess
test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)


# ✅ 모델 학습
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ✅ 평가
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Real", "Deep"]))
