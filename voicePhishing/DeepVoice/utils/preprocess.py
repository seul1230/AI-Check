import librosa
import numpy as np

import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_audio(path, min_length=1000):
    try:
        y, sr = sf.read(path)  # librosa보다 빠름
        if len(y) <= min_length or np.isnan(y).any():
            return path, False
        return path, True
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return path, False

def filter_nan_audio(real_voice_paths, deep_voice_paths):
    clean_real, removed_real = [], []
    clean_deep, removed_deep = [], []

    print("⏳ Real Voice 파일 검사 중...")
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(check_audio, path) for path in real_voice_paths]
        for future in as_completed(futures):
            path, is_valid = future.result()
            if is_valid:
                clean_real.append(path)
            else:
                print(f"⚠️ [REAL] 제거됨: {path}")
                removed_real.append(path)

    print("⏳ Deep Voice 파일 검사 중...")
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(check_audio, path) for path in deep_voice_paths]
        for future in as_completed(futures):
            path, is_valid = future.result()
            if is_valid:
                clean_deep.append(path)
            else:
                print(f"⚠️ [DEEP] 제거됨: {path}")
                removed_deep.append(path)

    print("✅ 필터링 완료")
    print(f"📊 남은 Real Voice: {len(clean_real)}")
    print(f"📊 남은 Deep Voice: {len(clean_deep)}")
    print(f"❌ 제거된 Real Voice: {len(removed_real)}")
    print(f"❌ 제거된 Deep Voice: {len(removed_deep)}")

    return (
        np.array(clean_real), np.array(clean_deep),
        removed_real, removed_deep
    )
