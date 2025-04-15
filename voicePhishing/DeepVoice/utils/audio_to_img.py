import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# 1. 음성 데이터 전처리 및 특징 추출

def load_audio(file_path, sr=16000):
    signal, sr = librosa.load(file_path, sr=sr)
    return signal, sr

def pre_emphasis(signal, coefficient=0.97):
    if len(signal) == 0:
        return signal  # 빈 배열이면 그대로 반환
    emphasized_signal = np.append(signal[0], signal[1:] - coefficient * signal[:-1])
    return emphasized_signal

def extract_mel_spectrogram(signal, sr, n_fft=400, hop_length=160, n_mels=128):
    # Mel-spectrogram 추출 후 로그 스케일 적용
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft,
                                              hop_length=hop_length, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

def extract_mfcc(signal, sr, n_mfcc=100, n_fft=400, hop_length=160):
    # MFCC 특징 100개 추출 (논문과 동일)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=n_fft, hop_length=hop_length)
    return mfccs

def save_features(mel_spec, mfcc, sr, output_dir, base_filename):
    """
    - Mel-Spectrogram은 흑백 이미지(.png)로 저장
    - MFCC는 수치 배열(.npy)로 저장 (BiLSTM 입력용으로 전치하여 (time, n_mfcc) 형태로 저장)
    """
    mel_output_path = os.path.join(output_dir, f"{base_filename}_mel.png")
    mfcc_output_path = os.path.join(output_dir, f"{base_filename}_mfcc.npy")

    if os.path.exists(mel_output_path) and os.path.exists(mfcc_output_path):
        print(f"Already processed: {base_filename}, skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Mel-Spectrogram 저장 (흑백 colormap 적용)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec, sr=sr, hop_length=160, x_axis='time', y_axis='mel', cmap='gray')
    plt.title('Mel-Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(mel_output_path)
    plt.close()

    # MFCC 배열은 npy 파일로 저장 (전치하여 (t, n_mfcc) 형태로)
    mfcc = mfcc.T
    np.save(mfcc_output_path, mfcc)

    print(f"Processed: {base_filename}")

if __name__ == '__main__':
    base_data_dir = r"C:\Users\SSAFY\autoML\data"
    output_directory = r"C:\Users\SSAFY\autoML\output"

    subfolders = ["tmp"]

    for subfolder in subfolders:
        data_folder = os.path.join(base_data_dir, subfolder)
        sub_output_dir = os.path.join(output_directory, subfolder)
        os.makedirs(sub_output_dir, exist_ok=True)

        # 재귀 탐색
        for root, dirs, files in os.walk(data_folder):
            for filename in files:
                if filename.lower().endswith('.wav'):
                    file_path = os.path.join(root, filename)
                    signal, sr = load_audio(file_path, sr=16000)
                    # 빈 신호라면 건너뜁니다.
                    if len(signal) == 0:
                        print(f"Empty signal in file: {file_path}, skipping.")
                        continue
                    emphasized_signal = pre_emphasis(signal)
                    mel_spec = extract_mel_spectrogram(emphasized_signal, sr)
                    mfcc = extract_mfcc(emphasized_signal, sr)
                    base_filename = os.path.splitext(filename)[0]
                    final_filename = f"{subfolder}_{base_filename}"
                    save_features(mel_spec, mfcc, sr, sub_output_dir, final_filename)
