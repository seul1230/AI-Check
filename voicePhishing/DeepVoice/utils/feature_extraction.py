import tensorflow as tf  
import numpy as np       

def get_mel_sequence_tf(waveform, sr=16000, n_mels=128, frame_length=512, frame_step=256):
    stft = tf.signal.stft(waveform, frame_length=frame_length, frame_step=frame_step)
    spectrogram = tf.abs(stft)

    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=sr
    )
    mel_spec = tf.tensordot(spectrogram, mel_weights, axes=1)
    mel_spec = tf.math.log(mel_spec + 1e-6)

    return mel_spec  # (time, n_mels) ✅ No transpose here!

def get_vector_features_tf(waveform):
    """
    간단한 통계 feature 추출 (TensorFlow만 사용)
    - return: shape (10,) 벡터
    """
    energy = tf.reduce_mean(tf.square(waveform))
    zcr = tf.reduce_mean(tf.cast(tf.math.not_equal(waveform[1:] * waveform[:-1], 0), tf.float32))

    max_amp = tf.reduce_max(tf.abs(waveform))
    mean_amp = tf.reduce_mean(waveform)
    std_amp = tf.math.reduce_std(waveform)

    # zero padding으로 차원 맞추기
    return tf.stack([energy, zcr, max_amp, mean_amp, std_amp] + [0.0] * 5)  # (10,)



# def get_mel_sequence_tf(waveform, sr=16000, n_mels=128, frame_length=512, frame_step=256):
#     """
#     BiLSTM 입력용 mel 시퀀스: shape = (time, n_mels)
#     """
#     stft = tf.signal.stft(waveform, frame_length=frame_length, frame_step=frame_step)
#     spectrogram = tf.abs(stft)

#     mel_weights = tf.signal.linear_to_mel_weight_matrix(
#         num_mel_bins=n_mels,
#         num_spectrogram_bins=spectrogram.shape[-1],
#         sample_rate=sr
#     )
#     mel_spec = tf.tensordot(spectrogram, mel_weights, axes=1)
#     mel_spec = tf.math.log(mel_spec + 1e-6)

#     return mel_spec  # (time, n_mels)


def get_spectrogram(waveform, sr=16000, max_frames=500):
    # STFT
    stft = tf.signal.stft(waveform, frame_length=1024, frame_step=256)
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

def get_mel_spectrogram_tf(waveform, sr=16000, n_mels=128, frame_length=512, frame_step=256):
    """
    TensorFlow 기반 Mel-Spectrogram 추출 함수
    - waveform: 1D float32 tensor
    - return: (n_mels, time, 1)
    """
    stft = tf.signal.stft(waveform, frame_length=frame_length, frame_step=frame_step)
    spectrogram = tf.abs(stft)

    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=sr
    )
    mel_spec = tf.tensordot(spectrogram, mel_weights, axes=1)
    mel_spec = tf.math.log(mel_spec + 1e-6)  # log-scale

    mel_spec = tf.expand_dims(tf.transpose(mel_spec), -1)  # (time, n_mels) → (n_mels, time, 1)
    return mel_spec

# ✅ 추가: mel + mfcc + pitch 조합
def get_combined_features(waveform, sr=16000, max_frames=500):
    stft = tf.signal.stft(waveform, frame_length=512, frame_step=256)
    spectrogram = tf.abs(stft)

    # mel
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

    # mfcc
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :13]

    # pitch (spectral centroid 기반 근사)
    centroid = tf.reduce_sum(
        tf.cast(tf.range(num_spectrogram_bins), tf.float32)[tf.newaxis, :] * spectrogram,
        axis=-1
    ) / (tf.reduce_sum(spectrogram, axis=-1) + 1e-6)
    pitch = tf.expand_dims(centroid, -1)

    # pad or trim
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


import librosa
import numpy as np

def get_advanced_145_features(audio, sr=16000, max_len=500):
    """
    딥페이크 탐지를 위한 고급 오디오 특성 조합
    - log-mel spectrogram
    - MFCC
    - Pitch (f0)
    - Zero Crossing Rate
    - Spectral Centroid
    - Spectral Bandwidth
    """
    # 1. Log-Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=512, hop_length=160, n_mels=128)
    log_mel = librosa.power_to_db(mel_spec).astype(np.float32)

    # 2. MFCC
    mfcc = librosa.feature.mfcc(S=log_mel, sr=sr, n_mfcc=13)

    # 3. Pitch (f0)
    f0, _, _ = librosa.pyin(audio, sr=sr, frame_length=1024, hop_length=160, fmin=50, fmax=500)
    f0 = np.nan_to_num(f0, nan=0.0).reshape(1, -1)  # (1, T)

    # 4. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=512, hop_length=160)  # (1, T)

    # 5. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=512, hop_length=160)

    # 6. Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=512, hop_length=160)

    # 🔄 시간 정렬 (모두 동일한 길이로 padding or trimming)
    def pad_or_trim(x, target_len=max_len):
        if x.shape[1] < target_len:
            return np.pad(x, ((0, 0), (0, target_len - x.shape[1])), mode='constant')
        return x[:, :target_len]

    log_mel = pad_or_trim(log_mel)
    mfcc = pad_or_trim(mfcc)
    f0 = pad_or_trim(f0)
    zcr = pad_or_trim(zcr)
    spectral_centroid = pad_or_trim(spectral_centroid)
    spectral_bandwidth = pad_or_trim(spectral_bandwidth)

    # 📦 최종 결합 (Feature 차원 기준으로 concat)
    combined = np.concatenate([
        log_mel,               # (128, T)
        mfcc,                 # (13, T)
        f0,                   # (1, T)
        zcr,                  # (1, T)
        spectral_centroid,    # (1, T)
        spectral_bandwidth    # (1, T)
    ], axis=0)  # shape: (n_features, T)

    # 🧊 채널 차원 추가
    combined = combined[..., np.newaxis]  # shape: (n_features, T, 1)

    return combined.astype(np.float32)  # 최종 shape: (features, time, 1)

def get_advanced_145_features_tf(waveform, sr=16000, n_fft=512, hop_length=160, max_len=500):
    # 1. STFT → Spectrogram
    stft = tf.signal.stft(waveform, frame_length=n_fft, frame_step=hop_length, fft_length=n_fft)
    spectrogram = tf.abs(stft)  # (T, F)

    # 2. Mel-Spectrogram
    num_spectrogram_bins = spectrogram.shape[-1]
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=128,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sr,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0
    )
    mel_spectrogram = tf.matmul(spectrogram, mel_weight_matrix)  # (T, 128)
    log_mel = tf.math.log(tf.maximum(mel_spectrogram, 1e-6))

    # 3. MFCC (only first 13)
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :13]  # (T, 13)

    # 4. Pitch approximation via Spectral Centroid
    freqs = tf.linspace(0.0, sr / 2, num_spectrogram_bins)
    centroid = tf.reduce_sum(spectrogram * freqs, axis=-1) / (tf.reduce_sum(spectrogram, axis=-1) + 1e-6)
    pitch = tf.expand_dims(centroid, -1)  # (T, 1)

    # 5. ZCR
    shifted = tf.roll(waveform, shift=1, axis=0)
    zcr_bool = tf.math.not_equal(tf.sign(waveform), tf.sign(shifted))
    zcr_val = tf.reduce_mean(tf.cast(zcr_bool, tf.float32))
    zcr = tf.ones(tf.shape(pitch)) * zcr_val  # (T, 1)

    # 6. Spectral Bandwidth
    mean_freq = centroid
    freq_diff_sq = tf.square(freqs - tf.expand_dims(mean_freq, -1))
    bandwidth = tf.sqrt(tf.reduce_sum(freq_diff_sq * spectrogram, axis=-1) / (tf.reduce_sum(spectrogram, axis=-1) + 1e-6))
    bandwidth = tf.expand_dims(bandwidth, -1)  # (T, 1)

    # 패딩/자르기 함수
    def pad_or_trim(tensor):
        cur_len = tf.shape(tensor)[0]
        pad_len = tf.maximum(0, max_len - cur_len)
        tensor = tensor[:max_len]
        return tf.pad(tensor, [[0, pad_len], [0, 0]])

    # 각 피처 패딩
    log_mel = pad_or_trim(log_mel)
    mfcc = pad_or_trim(mfcc)
    pitch = pad_or_trim(pitch)
    zcr = pad_or_trim(zcr)
    bandwidth = pad_or_trim(bandwidth)

    # Spectral Centroid도 다시 정리
    spectral_centroid = pad_or_trim(pitch)  # pitch와 같은 값

    # 최종 결합 (T, features)
    combined = tf.concat([log_mel, mfcc, pitch, zcr, spectral_centroid, bandwidth], axis=-1)

    # (features, time, 1) 형태로 변환
    combined = tf.transpose(combined, [1, 0])
    combined = tf.expand_dims(combined, -1)

    return combined  # shape: (features, time, 1)


def get_advanced_features(waveform, sr=16000, max_frames=500):
    # stft = tf.signal.stft(waveform, frame_length=1024, frame_step=256)
    # W0000 00:00:1742968872.148891  352480 fft_ops.cc:552] The CUDA FFT plan cache capacity of 512 has been exceeded. This may lead to extra time being spent constantly creating new plans.
    stft = tf.signal.stft(waveform, fft_length=512, frame_length=512, frame_step=256)
    spectrogram = tf.abs(stft)

    # Mel Spectrogram
    num_spectrogram_bins = spectrogram.shape[-1]
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=256,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sr,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0
    )
    mel = tf.matmul(spectrogram, mel_weight_matrix)
    # log_mel = tf.math.log(mel + 1e-5)

    log_mel = tf.math.log(tf.maximum(mel, 1e-5))  # 혹은 clip


    # # MFCC + ΔMFCC + ΔΔMFCC
    # mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :13]
    # delta_mfcc = tf.pad(mfcc[1:] - mfcc[:-1], [[1, 0], [0, 0]])
    # delta2_mfcc = tf.pad(delta_mfcc[1:] - delta_mfcc[:-1], [[1, 0], [0, 0]])

    # MFCC + ΔMFCC + ΔΔMFCC
    if tf.shape(log_mel)[-1] < 512:
        # fallback: zero MFCC
        mfcc = tf.zeros([tf.shape(log_mel)[0], 13], dtype=tf.float32)
    else:
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :13]
    
    delta_mfcc = tf.pad(mfcc[1:] - mfcc[:-1], [[1, 0], [0, 0]])
    delta2_mfcc = tf.pad(delta_mfcc[1:] - delta_mfcc[:-1], [[1, 0], [0, 0]])


    # Pitch (spectral centroid) + ΔPitch
    centroid = tf.reduce_sum(
        tf.cast(tf.range(num_spectrogram_bins), tf.float32)[tf.newaxis, :] * spectrogram,
        axis=-1
    ) / (tf.reduce_sum(spectrogram, axis=-1) + 1e-6)
    delta_pitch = tf.pad(centroid[1:] - centroid[:-1], [[1, 0]])
    pitch = tf.expand_dims(centroid, -1)
    delta_pitch = tf.expand_dims(delta_pitch, -1)

    # ZCR (Zero-Crossing Rate)
    zcr_val = tf.reduce_mean(
        tf.cast(tf.math.not_equal(tf.sign(waveform[:-1]), tf.sign(waveform[1:])), tf.float32)
    )
    zcr = tf.ones((max_frames, 1)) * zcr_val  # 동일값 반복

    # Spectral Flux
    flux = tf.sqrt(tf.reduce_sum(tf.square(spectrogram[1:] - spectrogram[:-1]), axis=-1))
    flux = tf.pad(flux, [[1, 0]])
    flux = tf.expand_dims(flux, -1)

    # ⬛ Padding or Trimming
    def pad_or_trim(tensor):
        tensor = tensor[:max_frames, :]
        pad_len = tf.maximum(0, max_frames - tf.shape(tensor)[0])
        return tf.pad(tensor, [[0, pad_len], [0, 0]])

    mel = pad_or_trim(log_mel)          # (T, 128)
    mfcc = pad_or_trim(mfcc)            # (T, 13)
    delta_mfcc = pad_or_trim(delta_mfcc)# (T, 13)
    delta2_mfcc = pad_or_trim(delta2_mfcc) # (T, 13)
    pitch = pad_or_trim(pitch)          # (T, 1)
    delta_pitch = pad_or_trim(delta_pitch)  # (T, 1)
    zcr = pad_or_trim(zcr)              # (T, 1)
    flux = pad_or_trim(flux)            # (T, 1)

    # ⬛ Feature 조합
    combined = tf.concat([
        mel, mfcc, delta_mfcc, delta2_mfcc,
        pitch, delta_pitch, zcr, flux
    ], axis=-1)  # (T, 171)

    combined = tf.transpose(combined, [1, 0])     # (input, time)
    # print(f"[DEBUG] feature shape before expand: {combined.shape}")
    combined = tf.expand_dims(combined, -1)
    # print(f"[DEBUG] feature shape after expand: {combined.shape}")

    return combined