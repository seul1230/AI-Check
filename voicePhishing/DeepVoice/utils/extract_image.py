import os
import glob
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def batch_wav_to_mel_images(
    input_folder,
    output_folder,
    sr=16000,
    n_fft=400,
    hop_length=160,
    n_mels=128
):
    """
    input_folder 내 모든 .wav 파일을 멜 스펙트로그램 이미지로 변환하여
    output_folder에 저장합니다.
    
    Parameters:
    -----------
    input_folder  : str
        .wav 파일들이 들어있는 폴더 경로
    output_folder : str
        변환된 이미지(.png)를 저장할 폴더 경로
    sr            : int
        샘플링 레이트 (디폴트 16000)
    n_fft         : int
        STFT 윈도우 크기
    hop_length    : int
        STFT 수행 시 프레임 간 이동 크기
    n_mels        : int
        Mel 주파수 축 개수
    """
    # 결과를 저장할 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # 폴더 내 모든 wav 파일 탐색
    wav_files = glob.glob(os.path.join(input_folder, "*.wav"))
    
    for wav_path in wav_files:
        # 1) 오디오 불러오기
        y, _ = librosa.load(wav_path, sr=sr)
        
        # 2) Mel-Spectrogram 계산
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        # 로그 스케일(dB)로 변환
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 3) 파일명 세팅
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        output_image_path = os.path.join(output_folder, f"{base_name}_mel.png")
        
        # 4) 시각화하여 이미지 파일로 저장(플롯 표시 X)
        plt.figure(figsize=(3, 3))        # 그림 크기를 적절히 조절
        plt.axis("off")                  # 축 제거
        librosa.display.specshow(
            mel_spec_db,
            sr=sr,
            hop_length=hop_length,
            x_axis=None,
            y_axis=None
        )
        # bbox_inches와 pad_inches를 이용해 여백 최소화
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"Saved Mel-Spectrogram: {output_image_path}")


# 사용 예시
if __name__ == "__main__":
    input_folder = "path/to/wav_folder"      # 실제 입력 폴더
    output_folder = "path/to/output_folder"  # 실제 출력 폴더
    
    batch_wav_to_mel_images(
        input_folder=input_folder,
        output_folder=output_folder,
        sr=16000,
        n_fft=400,
        hop_length=160,
        n_mels=128
    )
    print("변환이 완료되었습니다.")
