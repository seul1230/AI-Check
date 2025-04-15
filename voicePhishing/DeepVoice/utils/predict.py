import tensorflow as tf
import numpy as np
import librosa
import os
import soundfile as sf
from utils.feature_extraction import get_spectrogram, get_combined_features, get_advanced_features, get_advanced_145_features, get_advanced_145_features_tf, get_mel_spectrogram_tf, get_vector_features_tf, get_mel_sequence_tf

def decode_audio_librosa(file_path, sr=16000):
    waveform, _ = librosa.load(file_path, sr=sr, mono=True)
    print(f"🎧 Python waveform length: {len(waveform)} samples")
    return tf.convert_to_tensor(waveform, dtype=tf.float32)

def decode_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, sr = tf.audio.decode_wav(audio_binary, desired_channels=1)
    return tf.cast(tf.squeeze(audio, axis=-1), tf.float32)

def predict_audio(file_path, model_path, threshold=0.5):
    try:
        waveform = decode_audio(file_path)
        feature = get_advanced_145_features_tf(waveform)
        feature = tf.expand_dims(feature, 0)  # (1, input, time, 1)
        
        model = tf.keras.models.load_model(model_path, compile=False)
        prediction = model.predict(feature, verbose=0)
        deepfake_prob = prediction[0][1]  # softmax에서 class 1 (Deepfake) 확률
        real_prob = prediction[0][0]

        if deepfake_prob >= threshold:
            label = "‼️ Deepfake Voice"
            predicted_class = 1
            confidence = deepfake_prob
        else:
            label = "✅ Real Voice"
            predicted_class = 0
            confidence = real_prob

        print(f"🎙 {file_path.split('/')[-1]} → Predicted: {label} (Confidence: {confidence:.2%})")

        return predicted_class, confidence
    except Exception as e:
        print(f"⚠ Error processing {file_path}: {e}")
        return None

def decode_image_tf(image_path, width, height):
    image_binary = tf.io.read_file(image_path)
    image = tf.io.decode_png(image_binary, channels=1)  # 흑백 이미지
    image = tf.image.resize(image, [height, width])     # 리사이즈
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0, 1] 정규화
    image.set_shape([height, width, 1])  # shape 명시 (128, 500, 1)
    return image

def split_into_segments(waveform, sr=16000, segment_duration=4.0, trim_margin_sec=0.0):
    segment_samples = int(sr * segment_duration)
    margin_samples = int(sr * trim_margin_sec)
    total_samples = waveform.shape[0]

    # 앞뒤 margin 자르기 (1초 기본)
    if total_samples > 3 * margin_samples:
        # waveform = trim_leading_trailing_silence(waveform)
        waveform = waveform[margin_samples:total_samples - margin_samples]
    else:
        # 너무 짧은 경우엔 잘라버리면 안 되니까 그대로 사용
        print("⚠️ Warning: Too short to trim margins. Using original waveform.")
    
    trimmed_samples = waveform.shape[0]

    segments = []
    for start in range(0, trimmed_samples, segment_samples):
        end = start + segment_samples
        segment = waveform[start:end]
        if tf.shape(segment)[0] >= 256:  # STFT 최소 조건
            segments.append(segment)

    return segments

# --------------------------------------------------------------

def is_flat_fake(prob_list):
    mean = np.mean(prob_list)
    std = np.std(prob_list)
    max_p = np.max(prob_list)

    # 완전 낮고 flat한데 max가 미묘하게 튄 경우
    if mean < 0.01 and std < 0.003 and max_p > 0.005:
        return True
    return False

def is_deepfake_by_segments(probs):
    mean = np.mean(probs)
    std = np.std(probs)
    max_p = np.max(probs)

    if mean > 0.4:
        return True
    if np.sum(np.array(probs) > 0.9) >= 2:
        return True
    if is_flat_fake(probs):  # 추가
        return True
    return False


def predict_multimodal_tflite_segmented(audio_path, image_path, interpreter, IMG_WIDTH=500, IMG_HEIGHT=128, threshold=0.4):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    waveform = decode_audio_librosa(audio_path)
    segments = split_into_segments(waveform)
    print("segment 개수: ", len(segments))

    segment_probs = []

    # 라벨 추정
    basename = os.path.basename(audio_path)

    label = basename.split()[0].lower()
    if "real" in label:
        true_label = "✅ Real"
    else:
        true_label = "‼️ Fake or Mixed"


    # ✅ 공통 입력 준비 (전체 waveform 기준)
    full_vector = get_vector_features_tf(waveform)
    full_sequence = get_mel_sequence_tf(waveform)
    full_sequence = pad_sequence(full_sequence)
    full_image = decode_image_tf(image_path, IMG_WIDTH, IMG_HEIGHT)

    full_vector = tf.expand_dims(full_vector, 0)
    full_sequence = tf.expand_dims(full_sequence, 0)
    full_image = tf.expand_dims(full_image, 0)

    # ✅ 세그먼트 기반 예측
    if len(segments) > 0:
        for seg in segments:
            seg_vector = get_vector_features_tf(seg)
            seg_sequence = get_mel_sequence_tf(seg)
            seg_sequence = pad_sequence(seg_sequence)

            seg_vector = tf.expand_dims(seg_vector, 0)
            seg_sequence = tf.expand_dims(seg_sequence, 0)

            interpreter.set_tensor(input_details[0]['index'], seg_sequence.numpy())
            interpreter.set_tensor(input_details[1]['index'], full_image.numpy())
            interpreter.set_tensor(input_details[2]['index'], seg_vector.numpy())

            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            if len(output.shape) == 2:
                segment_probs.append(float(output[0][1]))
            elif len(output.shape) == 1:
                segment_probs.append(float(output[1]))
            else:
                raise ValueError(f"Unexpected output shape: {output.shape}")


        # 기존 평균 기반 판단
        mean_prob = np.mean(segment_probs)
        is_deepfake_segment = mean_prob > threshold
        
        # 최종 판단 조건 업데이트
        is_deepfake_segment = is_deepfake_segment or is_deepfake_by_segments(segment_probs)

    else:
        print(f"⚠️ No segments found for [{basename}], skipping segment analysis.")
        mean_prob = None
        is_deepfake_segment = None

    # ✅ 전체 waveform 예측
    interpreter.set_tensor(input_details[0]['index'], full_sequence.numpy())
    interpreter.set_tensor(input_details[1]['index'], full_image.numpy())
    interpreter.set_tensor(input_details[2]['index'], full_vector.numpy())

    for i in interpreter.get_input_details():
        print(i['name'], i['shape'], i['dtype'])


    interpreter.invoke()
    full_output = interpreter.get_tensor(output_details[0]['index'])

    if len(full_output.shape) == 2:
        deepfake_prob_full = float(full_output[0][1])
    elif len(full_output.shape) == 1:
        deepfake_prob_full = float(full_output[1])
    else:
        raise ValueError(f"Unexpected output shape: {full_output.shape}")

    is_deepfake_full = deepfake_prob_full > threshold
    

    # ✅ 출력
    print(f"📞 [{basename}]")
    print(f">> True Label: {true_label}")
    print(f"🎧 Full Waveform Deep Voice Probability: {deepfake_prob_full:.4f}")
    print(f"🧠 Full Prediction: {'‼️ Deepfake Voice Detected' if is_deepfake_full else '✅ Real Voice'}")
    print(f"➡️ Segment-based Deep Voice Probabilities: {np.round(segment_probs, 4).tolist()}")

    if mean_prob is not None:
        print(f"📊 Mean Deep Voice Probability: {mean_prob:.4f}")
        print(f"🧠 Segment-based Prediction / Final: {'‼️ Deepfake Voice Detected' if is_deepfake_segment else '✅ Real Voice'}")

    else:
        print(f"📊 Segment-based Prediction: ❌ Skipped")
    # print()

    return {
        "basename": basename,
        "true_label": true_label,
        "segment_probs": segment_probs,
        "mean_segment_prob": mean_prob,
        "is_deepfake_segment": is_deepfake_segment,
        "deepfake_prob_full": deepfake_prob_full,
        "is_deepfake_full": is_deepfake_full
    }



# ✅ 오디오 + 이미지 기반 TFLite 예측 함수
def predict_multimodal_tflite(audio_path, image_path, interpreter, IMG_WIDTH=500, IMG_HEIGHT=128):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    waveform = decode_audio_librosa(audio_path)
    vector = get_vector_features_tf(waveform)
    sequence = get_mel_sequence_tf(waveform)
    sequence = pad_sequence(sequence)

    image = decode_image_tf(image_path, IMG_WIDTH, IMG_HEIGHT)

    # 입력 shape 조정
    image = tf.expand_dims(image, 0)      # (1, 128, 500, 1)
    vector = tf.expand_dims(vector, 0)    # (1, 10)
    sequence = tf.expand_dims(sequence, 0)  # (1, 400, 128)

    # 입력 설정
    interpreter.set_tensor(input_details[0]['index'], sequence.numpy())  # (1, 400, 128)
    interpreter.set_tensor(input_details[1]['index'], image.numpy())     # (1, 128, 500, 1)
    interpreter.set_tensor(input_details[2]['index'], vector.numpy())    # (1, 10)


    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# --------------------------------------------------------------



# def split_into_segments(waveform, sr=16000, segment_duration=3.0):
#     segment_samples = int(sr * segment_duration)
#     total_samples = waveform.shape[0]
    
#     segments = []
#     for start in range(0, total_samples, segment_samples):
#         end = start + segment_samples
#         segment = waveform[start:end]
#         if tf.shape(segment)[0] >= 256:  # STFT 최소 조건
#             segments.append(segment)
#     return segments

def trim_silence(audio, sr=16000, top_db=60):
    """
    앞뒤 무음 구간 제거

    Parameters:
    - audio (np.ndarray): 오디오 waveform
    - sr (int): 샘플링레이트
    - top_db (int): 얼마나 작은 음압까지 무음으로 판단할지 (낮을수록 민감)

    Returns:
    - trimmed_audio (np.ndarray): 무음 잘린 오디오
    """
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio

def is_anomalous_segment_distribution(probs, threshold_ratio=0.5, low_bound=0.1):
    probs = np.array(probs)
    below = np.sum(probs < low_bound)
    ratio = below / len(probs)
    variance = np.std(probs)
    return ratio >= threshold_ratio and variance < 0.05

def is_unstable_distribution(probs, high=0.8, low=0.2, threshold=0.7):
    max_gap = np.max(probs) - np.min(probs)
    return max_gap > threshold

def has_sustained_rise(probs, rise_threshold=0.4, sustain_threshold=0.5, min_sustain_len=1):
    diffs = np.diff(probs)
    for i in range(len(diffs)):
        if diffs[i] > rise_threshold:
            sustain = probs[i+1:i+1+min_sustain_len]
            if all(p >= sustain_threshold for p in sustain):
                return True
    return False

def is_suspiciously_flat(probs, std_threshold=0.00005, range_threshold=0.0005, avg_threshold=0.1, min_segments=8):
    std = np.std(probs)
    rng = np.max(probs) - np.min(probs)
    avg = np.mean(probs)
    return (
        len(probs) >= min_segments and  # segment 수가 많은 경우에만
        std < std_threshold and
        rng < range_threshold and
        avg < avg_threshold
    )

    
def is_short_and_suspiciously_flat(probs, min_len=2, max_len=3, std_threshold=0.00005, avg_upper=0.085):
    std = np.std(probs)
    avg = np.mean(probs)
    return (
        len(probs) == 2 and  # <=2로 제한
        std < std_threshold and
        avg < avg_upper
    )
    

def trim_leading_trailing_silence(y, sr=16000, top_db=100):
    """
    앞/뒤 무음만 제거 (중간 무음은 그대로 유지).
    """
    # 전체 트리밍 범위만 계산 (samples 기준)
    non_silent_intervals = librosa.effects.split(y, top_db=top_db)

    if len(non_silent_intervals) == 0:
        # 모든 것이 무음이면 그대로 반환
        return y

    start = non_silent_intervals[0][0]
    end = non_silent_intervals[-1][1]

    return y[start:end]





# =============================================

IMG_HEIGHT = 128
def decode_image(image_path, img_width=500):
    image_binary = tf.io.read_file(image_path)
    image = tf.io.decode_png(image_binary, channels=1)  # 흑백
    image = tf.image.resize(image, [IMG_HEIGHT, img_width])  # 리사이즈 추가!
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0, 1] 정규화
    return image


def preprocess_cnn(audio_path, image_path, label):
    waveform = decode_audio(audio_path)
    image = decode_image(image_path)
    vector = get_vector_features_tf(waveform)
    label = tf.one_hot(label, depth=2)
    return (image, vector), label


# def pad_sequence(tensor, max_len=MAX_LEN):
#     tensor = tensor[:max_len]
#     pad_len = max_len - tf.shape(tensor)[0]
#     return tf.pad(tensor, [[0, pad_len], [0, 0]])


def pad_sequence(sequence, target_len=400):
    """
    (time, n_mels) 시퀀스를 target_len에 맞게 zero-padding 또는 crop
    """
    current_len = sequence.shape[0]

    if current_len == target_len:
        return sequence

    elif current_len < target_len:
        pad_len = target_len - current_len
        padding = tf.zeros((pad_len, sequence.shape[1]), dtype=sequence.dtype)
        return tf.concat([sequence, padding], axis=0)

    else:
        return sequence[:target_len, :]  # crop

def preprocess_bilstm(audio_path, image_path, label):
    waveform = decode_audio(audio_path)
    image = decode_image(image_path)
    vector = get_vector_features_tf(waveform)
    sequence = get_mel_sequence_tf(waveform)
    sequence = pad_sequence(sequence)  # <- 여기 패딩 추가
    label = tf.one_hot(label, depth=2)
    return (image, vector, sequence), label

#  -------------------------

def final_decision_multimodal(probs, model_conf):

    mean = np.mean(probs)
    max_val = np.max(probs)
    strong_count = np.sum(probs > 0.9)
    mid_range_ratio = np.sum((probs >= 0.4) & (probs <= 0.6)) / len(probs)

    # -----------------
    # 🔴 Deepfake 판단 조건
    if mean > 0.8:
        return True
    if strong_count >= 3 and max_val > 0.95:
        return True
    if mean > 0.6 and strong_count >= 2 and mid_range_ratio < 0.3:
        return True

    # -----------------
    # ✅ Real 판단 조건
    if mean < 0.4:
        return False
    if mid_range_ratio >= 0.6 and strong_count <= 1:
        return False
    if strong_count == 0 and max_val < 0.7:
        return False
    # ✅ Real로 보정하는 조건 강화
    if strong_count <= 2 and mean < 0.7 and mid_range_ratio > 0.3:
        return False


    # -----------------
    return None  # 판단 유보 → 모델 결과 그대로



def predict_call_with_segments_multimodal(file_path, image_path, model, MAX_LEN):
    waveform = decode_audio_librosa(file_path)
    segments = split_into_segments(waveform)

    input_images = []
    input_vectors = []
    input_sequences = []

    for seg in segments:
        # 👇 segment 기준 feature 생성
        vector = get_vector_features_tf(seg)
        sequence = get_mel_sequence_tf(seg)
        sequence = pad_sequence(sequence, MAX_LEN)

        # 👇 공통 image는 그대로 사용
        image = decode_image(image_path)

        input_images.append(image)
        input_vectors.append(vector)
        input_sequences.append(sequence)

    # 👇 배치 구성
    input_images = tf.stack(input_images)
    input_vectors = tf.stack(input_vectors)
    input_sequences = tf.stack(input_sequences)

    # 👇 세그먼트별 예측 → 확률만 추출
    probs = model.predict([input_images, input_vectors, input_sequences], verbose=0)[:, 1]
    probs = np.array(probs)

    try:
        # 전체 오디오 기준 대표 feature도 한 번 예측
        full_waveform = decode_audio(file_path)
        vector = get_vector_features_tf(full_waveform)
        sequence = get_mel_sequence_tf(full_waveform)
        sequence = pad_sequence(sequence)
        image = decode_image(image_path)

        image = tf.expand_dims(image, 0)
        vector = tf.expand_dims(vector, 0)
        sequence = tf.expand_dims(sequence, 0)

        pred = model.predict([image, vector, sequence], verbose=0)[0]
        deepfake_prob = pred[1]
        real_prob = pred[0]
        model_conf = max(deepfake_prob, real_prob)

        print(f"🎙 {os.path.basename(file_path)} → Predicted: {'‼️ Deepfake Voice' if deepfake_prob > 0.5 else '✅ Real Voice'} (Confidence: {model_conf:.2%})")

        decision = final_decision_multimodal(probs, model_conf)
        if decision is None:
            final = '‼️ Deepfake Voice' if deepfake_prob > 0.5 else '✅ Real Voice'
        else:
            final = '‼️ Deepfake Voice' if decision else '✅ Real Voice'

        print(f"final >> {final}")
        print("label  :", os.path.basename(file_path).split()[0].lower())
        print(probs.tolist(), "\n")

    except Exception as e:
        print(f"⚠ Error processing {file_path}: {e}")


def predict_call_with_segments_by_mean(file_path, image_path, model, MAX_LEN=500):
    # model_path = "model/multi_cnn_bilstm_epoch_10_batch_32_final_2025-03-27_07-06-42.keras"
    waveform = decode_audio_librosa(file_path)
    segments = split_into_segments(waveform)

    input_images = []
    input_vectors = []
    input_sequences = []

    for seg in segments:
        vector = get_vector_features_tf(seg)
        sequence = get_mel_sequence_tf(seg)
        sequence = pad_sequence(sequence, MAX_LEN)
        image = decode_image(image_path, 500)

        input_images.append(image)
        input_vectors.append(vector)
        input_sequences.append(sequence)

    input_images = tf.stack(input_images)
    input_vectors = tf.stack(input_vectors)
    input_sequences = tf.stack(input_sequences)

    # 👇 세그먼트별 예측 확률 추출
    probs = model.predict([input_images, input_vectors, input_sequences], verbose=0)[:, 1]
    mean_prob = np.mean(probs)

    # 👇 full waveform 예측 (보조 정보 출력용)
    full_waveform = decode_audio(file_path)
    vector = get_vector_features_tf(full_waveform)
    sequence = get_mel_sequence_tf(full_waveform)
    sequence = pad_sequence(sequence, MAX_LEN)
    image = decode_image(image_path, MAX_LEN)

    image = tf.expand_dims(image, 0)
    vector = tf.expand_dims(vector, 0)
    sequence = tf.expand_dims(sequence, 0)

    pred = model.predict([image, vector, sequence], verbose=0)[0]
    deepfake_prob = pred[1]
    real_prob = pred[0]

    print(f"🎙 {os.path.basename(file_path)} \n→ Predicted: {'‼️ Deepfake Voice' if deepfake_prob > 0.5 else '✅ Real Voice'}", pred)
    print("label  :", os.path.basename(file_path).split()[0].lower())
    print(probs.tolist(), "\n")


def is_deepfake_advanced(probs, deepfake_prob, threshold=0.5):
    mean_prob = np.mean(probs)
    max_prob = np.max(probs)
    std_prob = np.std(probs)
    strong_count = np.sum(probs > 0.8)

    # 우선순위 1️⃣: mean 확실히 높음
    if mean_prob > threshold:
        return True

    # 2️⃣: 일부 segment가 확실히 높고 분산도 큼
    if strong_count >= 2 and max_prob > 0.9 and std_prob > 0.15:
        return True

    # 3️⃣: full confidence 자체가 매우 높음
    if deepfake_prob > 0.9:
        return True

    # 4️⃣: 평균은 낮지만 예측 확률이 매우 분산돼 있음 → 이상징후
    if std_prob > 0.3 and max_prob > 0.85:
        return True

    # default: real
    return False

def predict_call_with_segments_hybrid(file_path, image_path, model, MAX_LEN=400):
    waveform = decode_audio_librosa(file_path)
    segments = split_into_segments(waveform)

    input_images, input_vectors, input_sequences = [], [], []

    for seg in segments:
        vector = get_vector_features_tf(seg)
        sequence = get_mel_sequence_tf(seg)
        sequence = pad_sequence(sequence, MAX_LEN)
        image = decode_image(image_path, MAX_LEN)

        input_images.append(image)
        input_vectors.append(vector)
        input_sequences.append(sequence)

    input_images = tf.stack(input_images)
    input_vectors = tf.stack(input_vectors)
    input_sequences = tf.stack(input_sequences)

    probs = model.predict([input_images, input_vectors, input_sequences], verbose=0)[:, 1]
    mean_prob = np.mean(probs)

    # Full waveform prediction
    full_waveform = decode_audio(file_path)
    vector = get_vector_features_tf(full_waveform)
    sequence = get_mel_sequence_tf(full_waveform)
    sequence = pad_sequence(sequence, MAX_LEN)
    image = decode_image(image_path, MAX_LEN)

    image = tf.expand_dims(image, 0)
    vector = tf.expand_dims(vector, 0)
    sequence = tf.expand_dims(sequence, 0)

    pred = model.predict([image, vector, sequence], verbose=0)[0]
    deepfake_prob = pred[1]
    real_prob = pred[0]

    # ✨ 하이브리드 판단
    is_fake = is_deepfake_advanced(probs, deepfake_prob)
    final = '‼️ Deepfake Voice' if is_fake else '✅ Real Voice'

    print(f"🎙 {os.path.basename(file_path)} \n→ Predicted: {final} (Full Confidence: {max(deepfake_prob, real_prob):.2%})")
    print("label  :", os.path.basename(file_path).split()[0].lower())
    print(probs.tolist(), "\n")



# ============================================== 

def final_decision(probs, model_confidence,
                   soft_threshold=0.5,
                   strong_confidence=0.7,
                   sliding_window_size=2,
                   sliding_threshold=0.55):
    probs = np.array(probs)
    if len(probs) == 0:
        return False  # 빈 입력은 Real 처리

    # ✅ 모델 확신이 높으면 그냥 모델 결과 그대로 따름
    if model_confidence > 0.6:
        return None  # 후처리 판단 X

    soft_avg = np.mean(probs)
    median_prob = np.median(probs)
    max_prob = np.max(probs)
    std_prob = np.std(probs)
    last_prob = probs[-1]

    # ✅ 예외 (Real로 간주)
    def is_surely_real():
        return max_prob < 0.2 and std_prob < 0.01 and soft_avg < 0.15

    if is_surely_real():
        return False

    # ✅ 탐지 조건 (Fake로 간주)
    def has_single_spike():
        return max_prob >= 0.55 and np.sum(probs >= 0.15) == 1

    def last_spike_only():
        return probs[-1] >= 0.55 and np.all(probs[:-1] < 0.15)

    def has_sliding_spike():
        for i in range(len(probs) - sliding_window_size + 1):
            if np.mean(probs[i:i+sliding_window_size]) >= sliding_threshold:
                return True
        return False

    if (
        has_single_spike() or
        last_spike_only() or
        has_sliding_spike() or
        soft_avg >= soft_threshold or
        max_prob >= 0.6 or
        (soft_avg >= 0.2 and max_prob >= 0.5)
    ):
        return True

    return False



def predict_call_with_segments_plain(file_path, model):
    waveform = decode_audio_librosa(file_path)
    segments = split_into_segments(waveform)

    features = [get_advanced_145_features_tf(seg) for seg in segments]
    features = tf.convert_to_tensor(features)
    probs = model.predict(features, verbose=0)[:, 1]
    probs = np.array(probs)

    try:
        feature = get_advanced_145_features_tf(decode_audio(file_path))
        feature = tf.expand_dims(feature, 0)
        prediction = model.predict(feature, verbose=0)[0]
        deepfake_prob = prediction[1]
        real_prob = prediction[0]
        model_conf = max(deepfake_prob, real_prob)

        # 기본 출력
        print(f"🎙 {file_path.split('/')[-1]} → Predicted: {'‼️ Deepfake Voice' if deepfake_prob > 0.5 else '✅ Real Voice'} (Confidence: {model_conf:.2%})")

        # 후처리 판단 (confidence <= 0.6일 경우만)
        decision = final_decision(probs, model_conf)
        if decision is None:
            # 후처리 미적용 → 모델 결과 그대로
            final = '‼️ Deepfake Voice' if deepfake_prob > 0.5 else '✅ Real Voice'
        else:
            final = '‼️ Deepfake Voice' if decision else '✅ Real Voice'

        print(f"final >> {final}")
        print("label  :", file_path.split('/')[-1].split()[0].lower())
        print(probs.tolist(), "\n")

    except Exception as e:
        print(f"⚠ Error processing {file_path}: {e}")


    
def predict_call_with_segments_cnn_bigru(file_path, model,
                               soft_threshold=0.5,
                               strong_confidence=0.7,
                               uncertainty_range=(0.4, 0.6),
                               min_segments=3,
                               sliding_window_size=2,
                               sliding_threshold=0.55,
                               anomaly_ratio=0.75,
                               anomaly_var=0.04):

    # model_path = "model/VGG19_BiGRU_best_model_2025-03-26_12-15-51.keras"
    
    waveform = decode_audio_librosa(file_path)
    segments = split_into_segments(waveform)

    features = [get_advanced_145_features(seg) for seg in segments]
    features = tf.convert_to_tensor(features)  # shape: (N, H, W, C)
    probs = model.predict(features, verbose=0)[:, 1]  # softmax class 1

    probs = np.array(probs)
    num_segments = len(probs)
    median_prob = np.median(probs)
    soft_avg = np.mean(probs)
    strong_count = np.sum(probs >= strong_confidence)
    uncertain_range_hit = np.all((probs >= uncertainty_range[0]) & (probs <= uncertainty_range[1]))
    diffs = np.diff(probs)
    sharp_rise = np.any(diffs > 0.4)
    flat_pattern_detected = is_suspiciously_flat(probs)

    low_prob_ratio = np.sum(probs < 0.1) / len(probs)
    variance = np.std(probs)
    anomaly_detected = (low_prob_ratio >= anomaly_ratio and variance < anomaly_var)

    def is_safe_low_flat_real(probs, min_len=3, max_len=6, std_thresh=0.00005, avg_thresh=0.082, max_thresh=0.085):
        return (
            min_len <= len(probs) <= max_len and
            np.std(probs) < std_thresh and
            np.mean(probs) < avg_thresh and
            np.max(probs) < max_thresh
        )

    def is_single_spike(probs, spike_threshold=0.55, context_range=0.15):
        """
        하나만 튀는 값인 경우 (ex. 0.57, 나머지 전부 0.07대)
        """
        if len(probs) < 3:
            return False
        count_spikes = np.sum(probs >= spike_threshold)
        surrounding_low = np.sum(probs < context_range)
        return count_spikes == 1 and surrounding_low >= len(probs) - 1

    def is_only_last_high(probs, last_thresh=0.55, others_thresh=0.15):
        if len(probs) < 2:
            return False
        return probs[-1] >= last_thresh and np.all(probs[:-1] < others_thresh)

    def is_tail_spike(probs, threshold=0.55, ratio=0.2):
        tail_len = int(len(probs) * ratio)
        return np.mean(probs[-tail_len:]) >= threshold and np.mean(probs[:-tail_len]) < 0.2


    is_safe_real_flat = is_safe_low_flat_real(probs)

    sliding_hits = 0
    for i in range(len(probs) - sliding_window_size + 1):
        window_avg = np.mean(probs[i:i + sliding_window_size])
        if window_avg >= sliding_threshold:
            sliding_hits += 1

    last_prob = probs[-1] if len(probs) > 0 else 0.0

    is_deep = False

    # 예외 조건 (Real이라고 확신하는 케이스)
    # 예외로 Real 판단
    if is_safe_real_flat or is_single_spike(probs) or is_only_last_high(probs):
        is_deep = False
    else:
        # 최소 segment 수가 부족하면 예측 보류 (또는 real로 처리)
        if num_segments < min_segments:
            is_deep = False
        # 여러 조건을 결합: 
        # - 강한 segment가 하나라도 있거나
        # - soft_avg와 median이 모두 일정 수준 이상일 때 deep로 판단
        # - sliding window에서 지속적으로 높은 값이 나타날 때 deep로 판단
        elif strong_count >= 1:
            is_deep = True
        elif soft_avg >= 0.5 and median_prob >= 0.5:
            is_deep = True
        elif sliding_hits >= 1:
            is_deep = True
        # 마지막 segment만 높은 경우는 예외처리
        elif is_only_last_high(probs):
            is_deep = False
        else:
            is_deep = False

    print(f"📞 [{file_path.split('/')[-1]}] Segments: {num_segments}")
    print(f" - Soft Avg: {soft_avg:.2%}")
    print(f" - Strong Count (≥{strong_confidence}): {strong_count}")
    print(f" - Sliding Hits (avg ≥ {sliding_threshold}): {sliding_hits}")
    print(f" - Median Prob: {median_prob:.2%}")
    print(f" - Last Segment Prob: {last_prob:.2%}")
    print(f" - Anomalous Pattern Detected: {anomaly_detected}")

    try:
        threshold = 0.5
        waveform = decode_audio(file_path)
        feature = get_advanced_145_features(waveform)
        feature = tf.expand_dims(feature, 0)

        prediction = model.predict(feature, verbose=0)
        deepfake_prob = prediction[0][1]
        real_prob = prediction[0][0]

        if deepfake_prob >= threshold:
            label = "‼️ Deepfake Voice"
            predicted_class = 1
            confidence = deepfake_prob
        else:
            label = "✅ Real Voice"
            predicted_class = 0
            confidence = real_prob

        
        print(f"🎙 {file_path.split('/')[-1]} → Predicted: {label} (Confidence: {confidence:.2%})")
        complete_predicted, complete_confidence = predicted_class, confidence
        # is_deep = complete_predicted == 0 and predicted_class == 0
        print(f"🔍 Final Prediction: {'‼️ Deepfake Voice Call' if is_deep and not (complete_predicted == 0 and complete_confidence > 0.9) else '✅ Real Voice Call'}")
    except Exception as e:
        print(f"⚠ Error processing {file_path}: {e}")

    
    print(f"{'🅾️' if (file_path.split('/')[-1].split()[0].lower() == 'real') == (not is_deep) else '❌'} -> ✅ 정답: {'Real' if file_path.split('/')[-1].split()[0].lower() == 'real' else 'fake'} | 예측: {'Real' if not is_deep else 'Fake'}")

    print(probs.tolist(), "\n")
    return is_deep, soft_avg


# # =============================
def predict_call_with_segments_cnn_bilistm(file_path, model,
                               soft_threshold=0.5,
                               strong_confidence=0.7,
                               uncertainty_range=(0.4, 0.6),
                               min_segments=3,
                               sliding_window_size=2,
                               sliding_threshold=0.55,
                               anomaly_ratio=0.75,
                               anomaly_var=0.04):

    # model_path = "model/VGG19_BiLSTM_ensemble_best_model_2025-03-26_01-51-00.keras"
    
    waveform = decode_audio_librosa(file_path)
    segments = split_into_segments(waveform)

    features = [get_advanced_features(seg) for seg in segments]
    features = tf.convert_to_tensor(features)  # shape: (N, H, W, C)
    probs = model.predict(features, verbose=0)[:, 1]  # softmax class 1

    probs = np.array(probs)
    num_segments = len(probs)
    median_prob = np.median(probs)
    soft_avg = np.mean(probs)
    strong_count = np.sum(probs >= strong_confidence)
    uncertain_range_hit = np.all((probs >= uncertainty_range[0]) & (probs <= uncertainty_range[1]))
    diffs = np.diff(probs)
    sharp_rise = np.any(diffs > 0.4)
    flat_pattern_detected = is_suspiciously_flat(probs)

    low_prob_ratio = np.sum(probs < 0.1) / len(probs)
    variance = np.std(probs)
    anomaly_detected = (low_prob_ratio >= anomaly_ratio and variance < anomaly_var)

    def is_safe_low_flat_real(probs, min_len=3, max_len=6, std_thresh=0.00005, avg_thresh=0.082, max_thresh=0.085):
        return (
            min_len <= len(probs) <= max_len and
            np.std(probs) < std_thresh and
            np.mean(probs) < avg_thresh and
            np.max(probs) < max_thresh
        )

    is_safe_real_flat = is_safe_low_flat_real(probs)

    sliding_hits = 0
    for i in range(len(probs) - sliding_window_size + 1):
        window_avg = np.mean(probs[i:i + sliding_window_size])
        if window_avg >= sliding_threshold:
            sliding_hits += 1

    last_prob = probs[-1] if len(probs) > 0 else 0.0

    is_deep = False

    # 🧯 예외 처리 먼저
    if is_safe_real_flat:
        is_deep = False
    else:
        if strong_count >= 1:
            is_deep = True
        elif soft_avg >= soft_threshold:
            is_deep = True
        elif sliding_hits >= 1:
            is_deep = True
        elif last_prob >= 0.35:
            is_deep = True
        elif anomaly_detected:  # 예외 없이 그냥 탐지
            is_deep = True
        elif flat_pattern_detected:
            is_deep = True
        elif sharp_rise and soft_avg > 0.1:
            is_deep = True
        elif soft_avg >= 0.2:
            is_deep = True
        elif soft_avg >= 0.12 and median_prob >= 0.12:
            is_deep = True


    print(f"📞 [{file_path.split('/')[-1]}] Segments: {num_segments}")
    print(f" - Soft Avg: {soft_avg:.2%}")
    print(f" - Strong Count (≥{strong_confidence}): {strong_count}")
    print(f" - Sliding Hits (avg ≥ {sliding_threshold}): {sliding_hits}")
    print(f" - Median Prob: {median_prob:.2%}")
    print(f" - Last Segment Prob: {last_prob:.2%}")
    print(f" - Anomalous Pattern Detected: {anomaly_detected}")

    try:
        threshold = 0.5
        waveform = decode_audio(file_path)
        feature = get_advanced_features(waveform)
        feature = tf.expand_dims(feature, 0)

        prediction = model.predict(feature, verbose=0)
        deepfake_prob = prediction[0][1]
        real_prob = prediction[0][0]

        if deepfake_prob >= threshold:
            label = "‼️ Deepfake Voice"
            predicted_class = 1
            confidence = deepfake_prob
        else:
            label = "✅ Real Voice"
            predicted_class = 0
            confidence = real_prob

        
        print(f"🎙 {file_path.split('/')[-1]} → Predicted: {label} (Confidence: {confidence:.2%})")
        complete_predicted, complete_confidence = predicted_class, confidence
        print(f"🔍 Final Prediction: {'‼️ Deepfake Voice Call' if is_deep and not (complete_predicted == 0 and complete_confidence > 0.9) else '✅ Real Voice Call'}")
    except Exception as e:
        print(f"⚠ Error processing {file_path}: {e}")

    
    print(f"{'🅾️' if (file_path.split('/')[-1].split()[0].lower() == 'real') == (not is_deep) else '❌'} -> ✅ 정답: {'Real' if file_path.split('/')[-1].split()[0].lower() == 'real' else 'fake'} | 예측: {'Real' if not is_deep else 'Fake'}")

    print(probs.tolist(), "\n")
    return is_deep, soft_avg





    
# #  -================================= # model_path = "model/VGG19_BiLSTM_ensemble_best_model_2025-03-26_01-51-00.keras"
# def predict_call_with_segments_cnn_bilistm(file_path, model,
#                                soft_threshold=0.5,
#                                strong_confidence=0.7,
#                                uncertainty_range=(0.4, 0.6),
#                                min_segments=3,
#                                sliding_window_size=2,
#                                sliding_threshold=0.55,
#                                anomaly_ratio=0.75,
#                                anomaly_var=0.04):

#     # model_path = "model/VGG19_BiLSTM_ensemble_best_model_2025-03-26_01-51-00.keras"
    
#     waveform = decode_audio_librosa(file_path)
#     segments = split_into_segments(waveform)

#     features = [get_advanced_features(seg) for seg in segments]
#     features = tf.convert_to_tensor(features)  # shape: (N, H, W, C)
#     probs = model.predict(features, verbose=0)[:, 1]  # softmax class 1

#     probs = np.array(probs)
#     num_segments = len(probs)
#     median_prob = np.median(probs)
#     soft_avg = np.mean(probs)
#     strong_count = np.sum(probs >= strong_confidence)
#     uncertain_range_hit = np.all((probs >= uncertainty_range[0]) & (probs <= uncertainty_range[1]))
#     diffs = np.diff(probs)
#     sharp_rise = np.any(diffs > 0.4)
#     flat_pattern_detected = is_suspiciously_flat(probs)

#     low_prob_ratio = np.sum(probs < 0.1) / len(probs)
#     variance = np.std(probs)
#     anomaly_detected = (low_prob_ratio >= anomaly_ratio and variance < anomaly_var)

#     # # 🆕 추가: 짧지만 안정적인 real voice 판별 방지 조건
#     # def is_safe_low_flat_real(probs, min_len=2, max_len=5, std_thresh=0.0005, avg_thresh=0.1):
#     #     return (
#     #         len(probs) >= min_len and len(probs) <= max_len and
#     #         np.std(probs) < std_thresh and
#     #         np.mean(probs) < avg_thresh and
#     #         np.max(probs) < 0.35  # 결정적인 segment 없음
#     #     )
#     def is_safe_low_flat_real(probs, min_len=2, max_len=15, std_thresh=0.001, avg_thresh=0.12):
#         return (
#             min_len <= len(probs) <= max_len and
#             np.std(probs) < std_thresh and
#             np.mean(probs) < avg_thresh and
#             np.max(probs) < 0.35
#         )


#     is_safe_real_flat = is_safe_low_flat_real(probs)

#     sliding_hits = 0
#     for i in range(len(probs) - sliding_window_size + 1):
#         window_avg = np.mean(probs[i:i + sliding_window_size])
#         if window_avg >= sliding_threshold:
#             sliding_hits += 1

#     last_prob = probs[-1] if len(probs) > 0 else 0.0

#     is_deep = False

#     # 🧯 우선순위 예외 처리: 실제처럼 보이는 낮고 평탄한 확률 분포
#     if is_safe_real_flat:
#         is_deep = False
#     else:
#         if strong_count >= 1:
#             is_deep = True
#         elif soft_avg >= soft_threshold:
#             is_deep = True
#         elif sliding_hits >= 1:
#             is_deep = True
#         elif last_prob >= 0.35:
#             is_deep = True
#         elif anomaly_detected and soft_avg > 0.09:
#             is_deep = True
#         elif flat_pattern_detected:
#             is_deep = True
#         elif sharp_rise and soft_avg > 0.1:
#             is_deep = True
#         elif soft_avg >= 0.2:
#             is_deep = True
#         elif soft_avg >= 0.12 and median_prob >= 0.12:
#             is_deep = True

#     # 🧯 조건: 실제 음성처럼 보이는 경우는 예외 처리
#     if is_deep and is_safe_real_flat:
#         is_deep = False

#     print(f"📞 [{file_path.split('/')[-1]}] Segments: {num_segments}")
#     print(f" - Soft Avg: {soft_avg:.2%}")
#     print(f" - Strong Count (≥{strong_confidence}): {strong_count}")
#     print(f" - Sliding Hits (avg ≥ {sliding_threshold}): {sliding_hits}")
#     print(f" - Median Prob: {median_prob:.2%}")
#     print(f" - Last Segment Prob: {last_prob:.2%}")
#     print(f" - Anomalous Pattern Detected: {anomaly_detected}")

#     try:
#         threshold = 0.5
#         waveform = decode_audio(file_path)
#         feature = get_advanced_features(waveform)
#         feature = tf.expand_dims(feature, 0)

#         prediction = model.predict(feature, verbose=0)
#         deepfake_prob = prediction[0][1]
#         real_prob = prediction[0][0]

#         if deepfake_prob >= threshold:
#             label = "‼️ Deepfake Voice"
#             predicted_class = 1
#             confidence = deepfake_prob
#         else:
#             label = "✅ Real Voice"
#             predicted_class = 0
#             confidence = real_prob

#         print(f"🎙 {file_path.split('/')[-1]} → Predicted: {label} (Confidence: {confidence:.2%})")
#         complete_predicted, complete_confidence = predicted_class, confidence
#     except Exception as e:
#         print(f"⚠ Error processing {file_path}: {e}")

#     print(f"🔍 Final Prediction: {'‼️ Deepfake Voice Call' if is_deep and not (complete_predicted == 0 and complete_confidence > 0.9) else '✅ Real Voice Call'}")
#     print(f"{'🅾️' if (file_path.split('/')[-1].split()[0].lower() == 'real') == (not is_deep) else '❌'} -> ✅ 정답: {'Real' if file_path.split('/')[-1].split()[0].lower() == 'real' else 'fake'} | 예측: {'Real' if not is_deep else 'Fake'}")

#     print(probs.tolist(), "\n")
#     return is_deep, soft_avg





# # ===============
# def predict_call_with_segments_cnn_bilistm(file_path, model,
#                                soft_threshold=0.5,
#                                strong_confidence=0.7,
#                                uncertainty_range=(0.4, 0.6),
#                                min_segments=3,
#                                sliding_window_size=2,
#                                sliding_threshold=0.55,
#                                anomaly_ratio=0.75,
#                                anomaly_var=0.04):
    
#     waveform = decode_audio_librosa(file_path)
#     segments = split_into_segments(waveform)

#     # model_path = "model/VGG19_BiLSTM_ensemble_best_model_2025-03-25_16-07-09.keras" # -- 거의 다 잡힘
    
#     # ✅ 효율적 (batch inference)
#     features = [get_advanced_features(seg) for seg in segments]
#     features = tf.convert_to_tensor(features)  # shape: (N, H, W, C)
#     probs = model.predict(features, verbose=0)[:, 1]  # softmax class 1

#     # probs = []
#     # for seg in segments:
#     #     feature = get_advanced_features(seg)
#     #     feature = tf.expand_dims(feature, 0)
#     #     prediction = model.predict(feature, verbose=0)
#     #     probs.append(prediction[0][1])

#     probs = np.array(probs)
#     num_segments = len(probs)
#     median_prob = np.median(probs)
#     soft_avg = np.mean(probs)
#     strong_count = np.sum(probs >= strong_confidence)
#     uncertain_range_hit = np.all((probs >= uncertainty_range[0]) & (probs <= uncertainty_range[1]))
#     diffs = np.diff(probs)
#     sharp_rise = np.any(diffs > 0.4)
#     flat_pattern_detected = is_suspiciously_flat(probs)

#     # ✅ 개선된 anomaly 판단
#     low_prob_ratio = np.sum(probs < 0.1) / len(probs)
#     variance = np.std(probs)
#     anomaly_detected = (low_prob_ratio >= anomaly_ratio and variance < anomaly_var)
#     short_flat_detected = is_short_and_suspiciously_flat(probs)

#     # ✅ sliding window 판단 추가
#     sliding_hits = 0
#     for i in range(len(probs) - sliding_window_size + 1):
#         window_avg = np.mean(probs[i:i + sliding_window_size])
#         if window_avg >= sliding_threshold:
#             sliding_hits += 1

#     last_prob = probs[-1] if len(probs) > 0 else 0.0

#     # ✅ 최종 판단 기준
#     is_deep = False
#     if strong_count >= 1:
#         is_deep = True
#     elif soft_avg >= soft_threshold:
#         is_deep = True
#     elif sliding_hits >= 1:
#         is_deep = True
#     elif last_prob >= 0.35:
#         is_deep = True
#     elif anomaly_detected and soft_avg > 0.09:
#         is_deep = True
#     elif flat_pattern_detected:  # 길면서 너무 평탄한 경우
#         is_deep = True
#     elif sharp_rise and soft_avg > 0.1:
#         is_deep = True
#     elif soft_avg >= 0.2:
#         is_deep = True
#     elif soft_avg >= 0.12 and median_prob >= 0.12:
#         is_deep = True
#     # elif short_flat_detected:
#     #     is_deep = True

#     # ✅ 출력
#     print(f"📞 [{file_path.split('/')[-1]}] Segments: {num_segments}")
#     print(f" - Soft Avg: {soft_avg:.2%}")
#     print(f" - Strong Count (≥{strong_confidence}): {strong_count}")
#     print(f" - Sliding Hits (avg ≥ {sliding_threshold}): {sliding_hits}")
#     print(f" - Median Prob: {median_prob:.2f}")
#     print(f" - Last Segment Prob: {last_prob:.2f}")
#     print(f" - Anomalous Pattern Detected: {anomaly_detected}")

#     try:
#         threshold = 0.5
#         waveform = decode_audio(file_path)
#         feature = get_advanced_features(waveform)
#         feature = tf.expand_dims(feature, 0)  # (1, input, time, 1)
        
#         prediction = model.predict(feature, verbose=0)
#         deepfake_prob = prediction[0][1]  # softmax에서 class 1 (Deepfake) 확률
#         real_prob = prediction[0][0]

#         if deepfake_prob >= threshold:
#             label = "‼️ Deepfake Voice"
#             predicted_class = 1
#             confidence = deepfake_prob
#         else:
#             label = "✅ Real Voice"
#             predicted_class = 0
#             confidence = real_prob

#         print(f"🎙 {file_path.split('/')[-1]} → Predicted: {label} (Confidence: {confidence:.2%})")
#         complete_predicted, complete_confidence = predicted_class, confidence #predict_audio(file_path, model_path)
#     except Exception as e:
#         print(f"⚠ Error processing {file_path}: {e}")
    
#     # complete_predicted, complete_confidence = predict_audio(file_path, model_path)
#     print(f"🔍 Final Prediction: {'‼️ Deepfake Voice Call' if is_deep and not (complete_predicted == 0 and complete_confidence > 0.9) else '✅ Real Voice Call'}")
#     print(probs.tolist(), "\n")
#     return is_deep, soft_avg