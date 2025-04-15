import os

def load_audio_file_paths(relative_dir, exts=(".wav", ".mp3", ".flac")):
    """
    주어진 상대경로 내의 오디오 파일들을 모두 리스트로 불러옴

    Args:
        relative_dir (str): 기준이 되는 상대 경로 (예: 'dataset/audio/train')
        exts (tuple): 포함할 확장자들

    Returns:
        List[str]: 오디오 파일 경로 리스트
    """
    audio_paths = []

    for root, _, files in os.walk(relative_dir):
        for file in files:
            if file.lower().endswith(exts):
                full_path = os.path.join(root, file)
                audio_paths.append(full_path)

    return audio_paths

def load_img_file_paths(relative_dir, exts=(".png")):
    """
    주어진 상대경로 내의 오디오 파일들을 모두 리스트로 불러옴

    Args:
        relative_dir (str): 기준이 되는 상대 경로 (예: 'dataset/audio/train')
        exts (tuple): 포함할 확장자들

    Returns:
        List[str]: 오디오 파일 경로 리스트
    """
    image_paths = []

    for root, _, files in os.walk(relative_dir):
        for file in files:
            if file.lower().endswith(exts):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)

    return image_paths

    