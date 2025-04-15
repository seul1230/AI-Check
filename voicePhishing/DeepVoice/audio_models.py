import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import applications
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    Permute, Concatenate, RepeatVector, Resizing,
    BatchNormalization, Input, Bidirectional, LSTM, Reshape,
    GlobalAveragePooling2D, Average, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Layer, GRU
)

from keras.saving import register_keras_serializable

import tensorflow.keras.backend as K

def build_cnn(input_shape=128, dropout=0.5):
    """ CNN 모델 """
    input_layer = Input(shape=(input_shape, 500, 1))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_bilstm(input_shape=128, dropout=0.5):
    """ CNN + BiLSTM 모델 """
    input_layer = Input(shape=(input_shape, 500, 1))  # 128 (Mel bins), 500 (time frames), 1 (channel)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)  # CNN 출력 shape: (batch, ?, ?, 128)

    # ✅ CNN의 출력 크기를 동적으로 가져오기 (정적인 크기 사용)
    cnn_output_shape = K.int_shape(x)  # (batch, time_steps, feature_dim, channels)
    time_steps = cnn_output_shape[1]   # CNN 출력 후 남은 time_steps
    feature_dim = cnn_output_shape[2]  # CNN 출력 채널 (128)
    channels = cnn_output_shape[3]     # CNN 최종 채널 수

    # ✅ Reshape 적용 (올바른 feature_dim 사용)
    x = Reshape((time_steps, feature_dim * channels))(x)  # batch_size 유지, feature_dim * channels로 변환

    # BiLSTM 추가 (양방향 LSTM)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64))(x)  # 최종 시퀀스 출력

    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    #               loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_vgg19_bilstm(input_height=142, dropout_rate=0.3):
    """
    VGG19 기반 CNN + BiLSTM 모델 구성
    input_height: Mel + MFCC + Pitch의 총 feature 차원 수 (기본 128+13+1 = 142)
    """
    input_layer = Input(shape=(input_height, None, 1))  # (142, time, 1)

    # 1채널을 3채널로 확장 (VGG19는 3채널을 요구)
    x = Concatenate(axis=-1)([input_layer, input_layer, input_layer])  # (142, time, 3)

    # 시간축을 가로로 간주해서 VGG19 적용 -> weight=None으로 랜덤 초기화 (transfer learning은 적합하지 않음)
    # resize: VGG19는 최소 32x32 이상 필요하므로 time축이 작으면 오류 발생 → 사전 reshape 보장 필요
    x = Permute((2, 1, 3))(x)  # (time, 142, 3)
    x = Resizing(128, 128)(x)  # (224, 224, 3)

    # VGG19 가져오기
    vgg = applications.VGG19(include_top=False, weights=None, input_shape=(128, 128, 3))
    x = vgg(x)
    x = GlobalAveragePooling2D()(x)

    # BiLSTM 처리 (임베딩처럼 취급)
    x = RepeatVector(1)(x)  # (1, feature_dim)
    x = Bidirectional(LSTM(128, return_sequences=False))(x)

    # 분류기
    x = Dropout(dropout_rate)(x)
    output = Dense(2, activation='softmax')(x)

    return Model(inputs=input_layer, outputs=output)

def build_vgg19_bilstm_ensemble(input_shape=(299, None, 1), dropout_rate=0.3):
    """
    VGG19 기반 CNN과 BiLSTM 기반 모델을 앙상블(soft voting)한 모델
    - VGG19는 (142, time, 1) → (128, 128, 3) 로 reshape 후 사용
    - BiLSTM은 시계열 그대로 사용
    - 두 모델의 softmax 출력을 평균 (Soft Voting)
    """

    # ⬛ Input
    input_layer = Input(shape=input_shape)

    ### 🧠 VGG19 Branch ###
    x_vgg = Concatenate(axis=-1)([input_layer, input_layer, input_layer])  # (142, time, 3)
    x_vgg = Permute((2, 1, 3))(x_vgg)  # (time, 142, 3)
    x_vgg = Resizing(128, 128)(x_vgg)

    vgg = applications.VGG19(include_top=False, weights=None, input_shape=(128, 128, 3))
    x_vgg = vgg(x_vgg)
    x_vgg = GlobalAveragePooling2D()(x_vgg)
    x_vgg = RepeatVector(1)(x_vgg)
    x_vgg = Bidirectional(LSTM(128, return_sequences=False))(x_vgg)
    x_vgg = Dropout(dropout_rate)(x_vgg)
    out_vgg = Dense(2, activation='softmax')(x_vgg)

    ### 🧠 BiLSTM Branch ###
    x_rnn = Permute((2, 1, 3))(input_layer)  # (time, 142, 1)
    x_rnn = Reshape((-1, input_shape[0]))(x_rnn)  # (time, 142)
    x_rnn = Bidirectional(LSTM(128, return_sequences=True))(x_rnn)
    x_rnn = Bidirectional(LSTM(64))(x_rnn)
    x_rnn = Dropout(dropout_rate)(x_rnn)
    out_rnn = Dense(2, activation='softmax')(x_rnn)

    ### 🧠 Soft Voting 앙상블 ###
    averaged_output = Average()([out_vgg, out_rnn])

    model = Model(inputs=input_layer, outputs=averaged_output)
    return model


# def non_cudnn_bilstm(units, return_sequences):
#     return Bidirectional(LSTM(
#         units,
#         return_sequences=return_sequences,
#         activation='tanh',
#         recurrent_activation='sigmoid',
#         use_bias=True,
#         unroll=False,
#         recurrent_dropout=0.0
#     ))

def non_cudnn_bilstm(units, return_sequences):
    return Bidirectional(LSTM(
        units,
        return_sequences=return_sequences,
        activation='tanh',
        recurrent_activation='sigmoid',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        unit_forget_bias=True,
        dropout=0.0,
        recurrent_dropout=0.0,
        implementation=2  # ✅ CuDNN 비활성화 강제
    ))


def build_vgg19_bilstm_ensemble_cpu_lstm(input_shape=(299, None, 1), dropout_rate=0.3):
    """
    CuDNN 없이 일반 LSTM 기반으로 수정된 VGG19 + BiLSTM 앙상블 모델
    """

    input_layer = Input(shape=input_shape)

    ### 🧠 VGG19 Branch ###
    x_vgg = Concatenate(axis=-1)([input_layer, input_layer, input_layer])  # (299, None, 3)
    x_vgg = Permute((2, 1, 3))(x_vgg)  # (time, 299, 3)
    x_vgg = Resizing(128, 128)(x_vgg)

    vgg = applications.VGG19(include_top=False, weights=None, input_shape=(128, 128, 3))
    x_vgg = vgg(x_vgg)
    x_vgg = GlobalAveragePooling2D()(x_vgg)
    x_vgg = RepeatVector(1)(x_vgg)
    x_vgg = non_cudnn_bilstm(128, return_sequences=False)(x_vgg)
    x_vgg = Dropout(dropout_rate)(x_vgg)
    out_vgg = Dense(2, activation='softmax')(x_vgg)

    ### 🧠 BiLSTM Branch ###
    x_rnn = Permute((2, 1, 3))(input_layer)  # (time, 299, 1)
    x_rnn = Reshape((-1, input_shape[0]))(x_rnn)  # (time, 299)
    x_rnn = non_cudnn_bilstm(128, return_sequences=True)(x_rnn)
    x_rnn = non_cudnn_bilstm(64, return_sequences=False)(x_rnn)
    x_rnn = Dropout(dropout_rate)(x_rnn)
    out_rnn = Dense(2, activation='softmax')(x_rnn)

    ### 🧠 Soft Voting 앙상블 ###
    averaged_output = Average()([out_vgg, out_rnn])

    model = Model(inputs=input_layer, outputs=averaged_output)
    return model


@register_keras_serializable()
class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):  # <- 핵심
        super().__init__(**kwargs)  # <- trainable, dtype 등 처리
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.att.key_dim,
            "num_heads": self.att.num_heads,
            "ff_dim": self.ffn.layers[0].units,
            "rate": self.dropout1.rate,
        })
        return config



def build_cnn_transformer(input_shape=(128, 500, 1), dropout=0.3, num_heads=4, ff_dim=256):
    """
    CNN + Transformer 기반 딥보이스 탐지 모델
    """
    inputs = Input(shape=input_shape)

    # ✅ CNN Block
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)  # → (batch, t', f', 128)

    # # ✅ Reshape to (batch, time, embed_dim)
    # t_dim = tf.shape(x)[1]
    # f_dim = tf.shape(x)[2]
    # c_dim = x.shape[-1]
    # x = Reshape((-1, x.shape[2] * x.shape[3]))(x)  # Flatten (time, f'*c) → token sequence

    # ✅ 올바른 코드 (KerasTensor에서 정적 shape 추출)
    cnn_output_shape = K.int_shape(x)  # returns tuple like (batch_size, t', f', c)
    t_dim = cnn_output_shape[1]
    f_dim = cnn_output_shape[2]
    c_dim = cnn_output_shape[3]
    
    # reshape for Transformer input
    x = Reshape((t_dim, f_dim * c_dim))(x)

    # ✅ Transformer Block
    transformer_block = TransformerEncoderBlock(embed_dim=x.shape[-1], num_heads=num_heads, ff_dim=ff_dim, rate=dropout)
    x = transformer_block(x)

    x = GlobalAveragePooling1D()(x)  # sequence → vector

    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_vgg19_bigru_ensemble(input_shape=(171, None, 1), dropout_rate=0.3):
    """
    VGG19 기반 CNN과 BiGRU 기반 모델을 앙상블(soft voting)
    - VGG19는 시각적 특징
    - GRU는 시계열 패턴
    - 둘의 출력을 softmax 후 평균
    """

    # ⬛ Input
    input_layer = Input(shape=input_shape)  # (171, time, 1)

    ### 🧠 VGG19 Branch ###
    x_vgg = Concatenate(axis=-1)([input_layer, input_layer, input_layer])  # (171, time, 3)
    x_vgg = Permute((2, 1, 3))(x_vgg)  # (time, 171, 3)
    x_vgg = Resizing(128, 128)(x_vgg)

    vgg = applications.VGG19(include_top=False, weights=None, input_shape=(128, 128, 3))
    x_vgg = vgg(x_vgg)
    x_vgg = GlobalAveragePooling2D()(x_vgg)
    x_vgg = RepeatVector(1)(x_vgg)
    x_vgg = Bidirectional(GRU(128, return_sequences=False))(x_vgg)
    x_vgg = Dropout(dropout_rate)(x_vgg)
    out_vgg = Dense(2, activation='softmax')(x_vgg)

    ### 🧠 GRU Branch ###
    x_rnn = Permute((2, 1, 3))(input_layer)  # (time, 171, 1)
    x_rnn = Reshape((-1, input_shape[0]))(x_rnn)  # (time, 171)
    x_rnn = Bidirectional(GRU(128, return_sequences=True))(x_rnn)
    x_rnn = Bidirectional(GRU(64))(x_rnn)
    x_rnn = Dropout(dropout_rate)(x_rnn)
    out_rnn = Dense(2, activation='softmax')(x_rnn)

    ### 🧠 Soft Voting 앙상블 ###
    averaged_output = Average()([out_vgg, out_rnn])

    model = Model(inputs=input_layer, outputs=averaged_output)
    return model

def build_img_cnn_bigru_ensemble(input_shape=(128, 500, 1), dropout_rate=0.3):
    """
    이미지 기반 CNN + 시계열 기반 BiGRU 앙상블 모델
    - CNN은 시각적 특징 추출
    - GRU는 시계열 패턴 학습
    - 두 분기의 softmax 출력을 평균 (soft voting)
    """
    input_layer = Input(shape=input_shape)  # (128, 500, 1)

    ### 🧠 CNN Branch ###
    x_cnn = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = MaxPooling2D((2, 2))(x_cnn)

    x_cnn = Conv2D(64, (3, 3), padding='same', activation='relu')(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = MaxPooling2D((2, 2))(x_cnn)

    x_cnn = Conv2D(128, (3, 3), padding='same', activation='relu')(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = MaxPooling2D((2, 2))(x_cnn)

    x_cnn = GlobalAveragePooling2D()(x_cnn)
    x_cnn = Dense(128, activation='relu')(x_cnn)
    x_cnn = Dropout(dropout_rate)(x_cnn)
    out_cnn = Dense(2, activation='softmax')(x_cnn)

    ### 🧠 BiGRU Branch ###
    x_rnn = Permute((2, 1, 3))(input_layer)  # (time, features, 1)
    x_rnn = Reshape((-1, input_shape[0]))(x_rnn)  # (time, features)
    x_rnn = Bidirectional(GRU(128, return_sequences=True))(x_rnn)
    x_rnn = Bidirectional(GRU(64))(x_rnn)
    x_rnn = Dropout(dropout_rate)(x_rnn)
    out_rnn = Dense(2, activation='softmax')(x_rnn)

    ### 🧠 Soft Voting 앙상블 ###
    averaged_output = Average()([out_cnn, out_rnn])

    model = Model(inputs=input_layer, outputs=averaged_output)
    return model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization, Concatenate
)

def build_multifeature_cnn_model(image_shape=(128, 500, 1), vector_shape=(10,), dropout_rate=0.3):
    """
    ✅ 경량 CNN + 벡터 feature 병합 모델
    - 이미지 입력은 간단한 CNN 구조로 처리
    - 벡터 입력은 Dense 처리
    - 중간에서 feature를 concat하여 분류기로 전달
    """

    # 🟦 이미지 입력 분기 (경량 CNN)
    image_input = Input(shape=image_shape, name="image_input")
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(image_input)  # 채널 수 축소
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = GlobalAveragePooling2D()(x)

    # 🟧 벡터 입력 분기 (Dense)
    vector_input = Input(shape=vector_shape, name="vector_input")
    v = Dense(64, activation='relu')(vector_input)
    v = Dropout(dropout_rate)(v)

    # 🔗 병합
    x = Concatenate()([x, v])
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    # ✅ 출력층
    output = Dense(2, activation='softmax')(x)

    return Model(inputs=[image_input, vector_input], outputs=output)


def build_multifeature_cnn_bilstm_ensemble(image_shape=(128, 500, 1), vector_shape=(10,), dropout_rate=0.3):
    """
    ✅ 이미지(CNN) + 벡터(Dense) + 오디오 시계열(BiLSTM) 앙상블 모델
    - 이미지와 벡터는 CNN + Dense 구조로 병합
    - 오디오 시퀀스는 BiLSTM으로 처리
    - 두 분기 출력 평균 (Soft Voting)
    """

    # 🟦 이미지 입력 (CNN)
    image_input = Input(shape=image_shape, name="image_input")
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(image_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = GlobalAveragePooling2D()(x)

    # 🟧 벡터 입력 (Dense)
    vector_input = Input(shape=vector_shape, name="vector_input")
    v = Dense(64, activation='relu')(vector_input)
    v = Dropout(dropout_rate)(v)

    # 🔗 CNN + 벡터 병합
    combined = Concatenate()([x, v])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(dropout_rate)(combined)
    out_combined = Dense(2, activation='softmax')(combined)

    # 🎧 오디오 시퀀스 입력 (Mel+벡터 feature 시퀀스)
    # 가정: audio_seq_shape = (time, feature) 형태로 정규화됨
    audio_seq_input = Input(shape=(400, 128), name="audio_sequence_input")  # 예시
    rnn = Bidirectional(LSTM(64, implementation=1, return_sequences=True))(audio_seq_input)
    rnn = Bidirectional(LSTM(32, implementation=1))(rnn)
    rnn = Dropout(dropout_rate)(rnn)
    out_rnn = Dense(2, activation='softmax')(rnn)

    # 🧠 Soft voting 앙상블
    from tensorflow.keras.layers import Average
    final_output = Average()([out_combined, out_rnn])

    return Model(inputs=[image_input, vector_input, audio_seq_input], outputs=final_output)

def build_multifeature_cnn_bigru_ensemble(image_shape=(128, 500, 1), vector_shape=(10,), dropout_rate=0.3):
    """
    ✅ 이미지(CNN) + 벡터(Dense) + 오디오 시계열(BiLSTM) 앙상블 모델
    - 이미지와 벡터는 CNN + Dense 구조로 병합
    - 오디오 시퀀스는 BiLSTM으로 처리
    - 두 분기 출력 평균 (Soft Voting)
    """

    # 🟦 이미지 입력 (CNN)
    image_input = Input(shape=image_shape, name="image_input")
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(image_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = GlobalAveragePooling2D()(x)

    # 🟧 벡터 입력 (Dense)
    vector_input = Input(shape=vector_shape, name="vector_input")
    v = Dense(64, activation='relu')(vector_input)
    v = Dropout(dropout_rate)(v)

    # 🔗 CNN + 벡터 병합
    combined = Concatenate()([x, v])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(dropout_rate)(combined)
    out_combined = Dense(2, activation='softmax')(combined)

    # 🎧 오디오 시퀀스 입력 (Mel+벡터 feature 시퀀스)
    # 가정: audio_seq_shape = (time, feature) 형태로 정규화됨
    audio_seq_input = Input(shape=(400, 128), name="audio_sequence_input")  # 예시
    # rnn = Bidirectional(GRU(64, implementation=1, return_sequences=True))(audio_seq_input)
    # rnn = Bidirectional(GRU(32, implementation=1))(rnn)
    rnn = Bidirectional(GRU(64, return_sequences=True, reset_after=False, recurrent_activation='sigmoid'))(audio_seq_input)
    rnn = Bidirectional(GRU(32, reset_after=False, recurrent_activation='sigmoid'))(rnn)
    rnn = Dropout(dropout_rate)(rnn)
    out_rnn = Dense(2, activation='softmax')(rnn)

    # 🧠 Soft voting 앙상블
    from tensorflow.keras.layers import Average
    final_output = Average()([out_combined, out_rnn])

    return Model(inputs=[image_input, vector_input, audio_seq_input], outputs=final_output)

def build_multifeature_cnn_bilstm_concat_ensemble(image_shape=(128, 400, 1), vector_shape=(10,), dropout_rate=0.3):
    """
    ✅ 이미지(CNN) + 벡터(Dense) + 오디오 시퀀스(BiLSTM) 분기 결합 후,
    Concat하여 Dense로 최종 판단하는 앙상블 구조
    """

    # 🟦 이미지 입력 (CNN)
    image_input = Input(shape=image_shape, name="image_input")
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(image_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = GlobalAveragePooling2D()(x)

    # 🟧 벡터 입력 (Dense)
    vector_input = Input(shape=vector_shape, name="vector_input")
    v = Dense(64, activation='relu')(vector_input)
    v = Dropout(dropout_rate)(v)

    # 🔗 CNN + 벡터 병합
    combined = Concatenate()([x, v])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(dropout_rate)(combined)
    out_combined = Dense(32, activation='relu')(combined)  # 👉 Softmax 대신 중간 출력

    # 🎧 오디오 시퀀스 입력 (Mel 시퀀스)
    audio_seq_input = Input(shape=(400, 128), name="audio_sequence_input")
    # rnn = Bidirectional(LSTM(64, return_sequences=True))(audio_seq_input)
    # rnn = Bidirectional(LSTM(32))(rnn)
    # after (순수 TF 연산으로 변경)
    rnn = Bidirectional(LSTM(64, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', unroll=True))(audio_seq_input)
    rnn = Bidirectional(LSTM(32, activation='tanh', recurrent_activation='sigmoid', unroll=True))(rnn)
    rnn = Dropout(dropout_rate)(rnn)
    out_rnn = Dense(32, activation='relu')(rnn)  # 👉 Softmax 대신 중간 출력

    # 🧠 Concat 후 최종 판단
    merged = Concatenate()([out_combined, out_rnn])  # (64,)
    final_output = Dense(2, activation='softmax')(merged)

    return Model(inputs=[image_input, vector_input, audio_seq_input], outputs=final_output)


def convert_model_to_tflite_with_dynamic_quantization(model):
    """
    동적 양자화를 사용하여 모델을 TensorFlow Lite 형식으로 변환하는 함수
    """
    # 모델을 TensorFlow Lite로 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 동적 양자화 설정
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    # 양자화된 모델로 변환
    tflite_model = converter.convert()

    return tflite_model