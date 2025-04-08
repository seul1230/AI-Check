
import librosa
librosa.cache = lambda func: func
import numpy as np


IMG_HEIGHT = 128
IMG_WIDTH = 500

# ----------------------------------------
# 🔉 오디오 로딩 (librosa 기반)
# ----------------------------------------
# def decode_audio_wave(file_path, sr=16000):
#     waveform, _ = librosa.load(file_path, sr=sr, mono=True)
#     return waveform

def decode_audio_wave(file_path, sr=16000):
    with wave.open(file_path, 'rb') as wf:
        num_channels = wf.getnchannels()
        framerate = wf.getframerate()
        num_frames = wf.getnframes()
        audio = wf.readframes(num_frames)
        waveform = np.frombuffer(audio, dtype=np.int16)
        if num_channels > 1:
            waveform = waveform[::num_channels]
        waveform = waveform.astype(np.float32) / 32768.0
        if framerate != sr:
            waveform = scipy.signal.resample_poly(waveform, sr, framerate)
        return waveform

# ----------------------------------------
# 📈 Feature 추출
# ----------------------------------------
def pre_emphasis(signal, coefficient=0.97):
    if len(signal) == 0:
        return signal
    return np.append(signal[0], signal[1:] - coefficient * signal[:-1])

def extract_log_mel_spec(signal, sr=16000, n_fft=400, hop_length=160, n_mels=128):
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft,
                                              hop_length=hop_length, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec  # shape: (128, time)

def create_mel_spectrogram_image(signal, sr=16000):
    emphasized = pre_emphasis(signal)
    mel_spec = extract_log_mel_spec(emphasized, sr)
    mel_spec -= mel_spec.min()
    mel_spec /= mel_spec.max() + 1e-6
    mel_spec = np.expand_dims(mel_spec, axis=-1)  # (128, 500, 1)
    return mel_spec.astype(np.float32)

def save_mel_spectrogram_image(signal, sr=16000, output_path="mel_output.png"):
    emphasized = pre_emphasis(signal)
    mel_spec = extract_log_mel_spec(emphasized, sr)
    mel_spec -= mel_spec.min()
    mel_spec /= mel_spec.max() + 1e-6
    mel_spec *= 255.0
    mel_spec = mel_spec.astype(np.uint8)
    img = Image.fromarray(mel_spec)
    img = img.convert("L")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img.save(output_path)
    return img, output_path

# ----------------------------------------
# 🔍 Segment, Feature, Padding
# ----------------------------------------
def split_into_segments(waveform, sr=16000, segment_duration=4.0):
    segment_samples = int(sr * segment_duration)
    total_samples = len(waveform)
    segments = []
    for start in range(0, total_samples, segment_samples):
        end = start + segment_samples
        segment = waveform[start:end]
        if len(segment) >= 256:
            segments.append(segment)
    return segments

def get_vector_features(waveform):
    energy = np.mean(np.square(waveform))
    zcr = np.mean(np.abs(np.diff(np.sign(waveform)))) / 2
    max_amp = np.max(np.abs(waveform))
    mean_amp = np.mean(waveform)
    std_amp = np.std(waveform)
    return [energy, zcr, max_amp, mean_amp, std_amp] + [0.0] * 5

def get_mel_sequence(waveform, sr=16000, n_mels=128, frame_length=512, frame_step=256):
    mel_spec = extract_log_mel_spec(waveform, sr=sr, n_fft=frame_length,
                                    hop_length=frame_step, n_mels=n_mels)
    return mel_spec.T  # shape: (time, 128)

def pad_sequence(seq, target_len=400):
    cur_len = seq.shape[0]
    if cur_len < target_len:
        pad = np.zeros((target_len - cur_len, seq.shape[1]), dtype=np.float32)
        seq = np.vstack([seq, pad])
    else:
        seq = seq[:target_len, :]
    return seq

# ----------------------------------------
# 🧩 전체 전처리 핵심 함수 (Kotlin에서 호출)
# ----------------------------------------
def extract_all_features(audio_path, mel_image_path, sequence_len=400):
    waveform = decode_audio_wave(audio_path)
    img, _ = save_mel_spectrogram_image(waveform, output_path=mel_image_path)

    # 전체 기준
    full_vector = get_vector_features(waveform)
    full_sequence = get_mel_sequence(waveform)
    full_sequence = pad_sequence(full_sequence, sequence_len)

    # 세그먼트 기준
    segments = split_into_segments(waveform)
    segment_vectors = []
    segment_sequences = []

    for seg in segments:
        v = get_vector_features(seg)
        s = get_mel_sequence(seg)
        s = pad_sequence(s, sequence_len)
        segment_vectors.append(v)
        segment_sequences.append(s)

    # mel image 디코딩
    mel_spec = np.array(img).astype(np.float32) / 255.0
    mel_spec = mel_spec.reshape((IMG_HEIGHT, IMG_WIDTH, 1))

    return {
        "full_vector": full_vector,
        "full_sequence": full_sequence.tolist(),
        "mel_image": mel_spec.tolist(),
        "segment_vectors": segment_vectors,
        "segment_sequences": [s.tolist() for s in segment_sequences]
    }

# ----------------------------------------
# 📣 Kotlin에서 호출할 함수
# ----------------------------------------
def process_audio_to_mel_and_features(audio_path, output_image_path):
    waveform = decode_audio_wave(audio_path)
    save_mel_spectrogram_image(waveform, output_path=output_image_path)

    full_vector = get_vector_features(waveform)
    full_sequence = pad_sequence(get_mel_sequence(waveform))

    segments = split_into_segments(waveform)
    segment_vectors = [get_vector_features(seg) for seg in segments]
    segment_sequences = [pad_sequence(get_mel_sequence(seg)) for seg in segments]

    return {
        "full_vector": full_vector,
        "full_sequence": full_sequence.tolist(),
        "segment_vectors": segment_vectors,
        "segment_sequences": [seq.tolist() for seq in segment_sequences]
    }
