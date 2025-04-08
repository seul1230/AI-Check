# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import wavfile
# from scipy.signal import spectrogram, resample
# import os
#
# def hz_to_mel(hz):
#     return 2595 * np.log10(1 + hz / 700)
#
# def mel_to_hz(mel):
#     return 700 * (10**(mel / 2595) - 1)
#
# def get_mel_filter(sr, n_fft=400, n_mels=128):
#     mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sr // 2), n_mels + 2)
#     hz_points = mel_to_hz(mel_points)
#     bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
#
#     filters = np.zeros((n_mels, n_fft // 2 + 1))
#     for i in range(1, len(bins) - 1):
#         l, c, r = bins[i - 1], bins[i], bins[i + 1]
#         if c - l > 0:
#             filters[i - 1, l:c] = (np.arange(l, c) - l) / (c - l)
#         if r - c > 0:
#             filters[i - 1, c:r] = (r - np.arange(c, r)) / (r - c)
#     return filters
#
# def save_mel_spectrogram_image(wav_path, out_path, sr_target=16000, n_fft=400, hop=160, n_mels=128):
#     sr, y = wavfile.read(wav_path)
#     if y.ndim > 1:
#         y = y[:, 0]  # mono
#     y = y.astype(np.float32) / 32768
#
#     if sr != sr_target:
#         y = resample(y, int(len(y) * sr_target / sr))
#         sr = sr_target
#
#     f, t, Sxx = spectrogram(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, mode='magnitude')
#     mel_filters = get_mel_filter(sr, n_fft=n_fft, n_mels=n_mels)
#     mel_spec = np.dot(mel_filters, Sxx)
#     log_mel = np.log10(mel_spec + 1e-6)
#
#     # 이미지 저장 (gray, no axes, no padding)
#     plt.figure(figsize=(10, 4))
#     plt.axis("off")
#     plt.imshow(log_mel, aspect='auto', origin='lower', cmap='gray')
#     plt.tight_layout(pad=0)
#     plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
#     plt.close()

# mel_generator_scipy.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, resample
import os

def pre_emphasis(signal, coeff=0.97):
    if len(signal) == 0:
        return signal
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

def get_mel_filter(sr, n_fft=400, n_mels=128):
    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sr // 2), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, len(bins) - 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        if c - l > 0:
            filters[i - 1, l:c] = (np.arange(l, c) - l) / (c - l)
        if r - c > 0:
            filters[i - 1, c:r] = (r - np.arange(c, r)) / (r - c)
    return filters

def save_mel_spectrogram_image(wav_path, out_path, sr_target=16000, n_fft=400, hop=160, n_mels=128):
    sr, y = wavfile.read(wav_path)
    if y.ndim > 1:
        y = y[:, 0]
    y = y.astype(np.float32) / 32768.0
    y = pre_emphasis(y)

    if sr != sr_target:
        y = resample(y, int(len(y) * sr_target / sr))
        sr = sr_target

    f, t, Sxx = spectrogram(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, mode='magnitude')
    mel_filters = get_mel_filter(sr, n_fft=n_fft, n_mels=n_mels)
    mel_spec = np.dot(mel_filters, Sxx)
    log_mel = np.log10(mel_spec + 1e-6)

    plt.figure(figsize=(10, 4))
    plt.axis("off")
    plt.imshow(log_mel, aspect='auto', origin='lower', cmap='gray')
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
