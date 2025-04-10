import os
os.environ["LIBROSA_CACHE_DIR"] = ""
os.environ["LIBROSA_CACHE_DISABLE"] = "1"

import joblib
class FakeMemory:
    def __init__(self, *args, **kwargs):
        self.location = None
    def cache(self, func): return func
    def clear(self, *args, **kwargs): pass
joblib.Memory = FakeMemory

import librosa
import tensorflow as tf
import matplotlib.pyplot as plt

# if not os.path.exists("/data/data/com.aicheck/files/chaquopy/installed_flag.txt"):
#     print("🔧 First-time Python setup…")
#     import librosa
#     import tensorflow as tf
#     import matplotlib.pyplot as plt
#     with open("/data/data/com.aicheck/files/chaquopy/installed_flag.txt", "w") as f:
#         f.write("done")
# else:
#     print("✅ Already installed, skipping re-import.")

import numpy as np

IMG_HEIGHT = 128
IMG_WIDTH = 500

def decode_audio_wave(file_path, sr=16000):
    waveform, _ = librosa.load(file_path, sr=sr, mono=True)
    return waveform

def pre_emphasis(signal, coefficient=0.97):
    if len(signal) == 0:
        return signal
    emphasized_signal = np.append(signal[0], signal[1:] - coefficient * signal[:-1])
    return emphasized_signal

def extract_log_mel_spec_tf(signal, sr=16000, n_fft=512, hop_length=256, n_mels=128):
    stft = tf.signal.stft(signal, frame_length=n_fft, frame_step=hop_length)
    spectrogram = tf.abs(stft)
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=stft.shape[-1],
        sample_rate=sr
    )
    mel_spec = tf.tensordot(spectrogram, mel_weights, axes=1)
    mel_spec = tf.math.log(tf.maximum(mel_spec, 1e-6))
    return mel_spec

def save_mel_spectrogram_image(wav_path, out_path,
                               sr=16000, n_fft=400, hop_length=160, n_mels=128, pre_emphasis_coef=0.97):
    import matplotlib
    matplotlib.use('Agg')
    import librosa.display
    y, _ = librosa.load(wav_path, sr=sr)
    if len(y) == 0:
        print(f"[Warning] Empty audio: {wav_path}")
        return
    y_emphasized = pre_emphasis(y, pre_emphasis_coef)
    mel_spec = librosa.feature.melspectrogram(y=y_emphasized, sr=sr, n_fft=n_fft,
                                              hop_length=hop_length, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spec, sr=sr, hop_length=hop_length, cmap='gray')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[OK] Saved mel image → {out_path}")

def split_into_segments(waveform, sr=16000, segment_duration=4.0):
    segment_samples = int(sr * segment_duration)
    total_samples = tf.shape(waveform)[0]
    segments = []
    for start in range(0, total_samples, segment_samples):
        end = start + segment_samples
        segment = waveform[start:end]
        if tf.shape(segment)[0] >= 256:
            segments.append(segment)
    return segments

def get_vector_features_tf(waveform):
    energy = tf.reduce_mean(tf.square(waveform))
    zcr = tf.reduce_mean(tf.cast(tf.math.not_equal(waveform[1:] * waveform[:-1], 0), tf.float32))
    max_amp = tf.reduce_max(tf.abs(waveform))
    mean_amp = tf.reduce_mean(waveform)
    std_amp = tf.math.reduce_std(waveform)
    return tf.stack([energy, zcr, max_amp, mean_amp, std_amp] + [0.0] * 5)

def get_mel_sequence_tf(waveform, sr=16000, n_mels=128, frame_length=512, frame_step=256):
    return extract_log_mel_spec_tf(waveform, sr, frame_length, frame_step, n_mels)

def pad_sequence(seq, target_len=400):
    cur_len = tf.shape(seq)[0]
    pad_len = tf.maximum(0, target_len - cur_len)
    padded = tf.pad(seq, [[0, pad_len], [0, 0]], constant_values=0.0)
    return padded[:target_len]

def decode_image_tf(image_path, width=500, height=128):
    image_binary = tf.io.read_file(image_path)
    image = tf.io.decode_png(image_binary, channels=1)
    image = tf.image.resize(image, [height, width])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape([height, width, 1])
    return image

def is_suspiciously_flat(prob_list):
    mean = np.mean(prob_list)
    std = np.std(prob_list)
    max_p = np.max(prob_list)

    # 평탄한데 평균이 꽤 높은 경우 (실제 fake가 섞인 mixed 음성일 수 있음)
    if mean > 0.26 and mean < 0.32 and std < 0.002:
        return True

    # 완전 낮은 확률인데 max만 튀는 기존 조건도 유지
    if mean < 0.01 and std < 0.003 and max_p > 0.005:
        return True

    return False


def is_deepfake_by_segments(probs):
    mean = np.mean(probs)
    if mean > 0.4:
        return True
    if np.sum(np.array(probs) > 0.9) >= 2:
        return True
    if is_suspiciously_flat(probs):  # <-- 여기에 새 기준 반영
        return True
    return False


def process_audio_to_mel_and_features(audio_path, output_image_path):
    waveform = decode_audio_wave(audio_path)
    if not isinstance(waveform, tf.Tensor):
        waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

    save_mel_spectrogram_image(audio_path, output_image_path)
    mel_tensor = decode_image_tf(output_image_path, IMG_WIDTH, IMG_HEIGHT)

    full_vector = get_vector_features_tf(waveform).numpy().tolist()
    full_sequence = pad_sequence(get_mel_sequence_tf(waveform)).numpy().tolist()

    segments = split_into_segments(waveform)
    segment_vectors = [get_vector_features_tf(seg).numpy().tolist() for seg in segments]
    segment_sequences = [pad_sequence(get_mel_sequence_tf(seg)).numpy().tolist() for seg in segments]

    return {
        "full_vector": full_vector,
        "full_sequence": full_sequence,
        "mel_image": mel_tensor.numpy().tolist(),
        "segment_vectors": segment_vectors,
        "segment_sequences": segment_sequences
    }
