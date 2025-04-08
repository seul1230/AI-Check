import os
os.environ["LIBROSA_CACHE_DIR"] = ""
os.environ["LIBROSA_CACHE_DISABLE"] = "1"

# joblib.Memory 가 file I/O 못 하게 mocking
import joblib
class FakeMemory:
    def __init__(self, *args, **kwargs):
        self.location = None  # 이거 꼭 있어야 함
    def cache(self, func): return func
    def clear(self, *args, **kwargs): pass
joblib.Memory = FakeMemory  # 반드시 librosa import 전에 실행

# 📦 이제 librosa import
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def save_mel_spectrogram_image(wav_path, out_path,
                               sr=16000, n_fft=400, hop_length=160, n_mels=128, pre_emphasis_coef=0.97):
    # 1. 로드
    y, sr_orig = librosa.load(wav_path, sr=sr)
    if len(y) == 0:
        print(f"[Warning] Empty audio: {wav_path}")
        return

    # 2. Pre-emphasis
    y_emphasized = np.append(y[0], y[1:] - pre_emphasis_coef * y[:-1])

    # 3. Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y_emphasized, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # 4. 저장 (gray scale, no axis, no padding)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spec, sr=sr, hop_length=hop_length, x_axis=None, y_axis=None, cmap='gray')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"[OK] Saved mel image → {out_path}")
