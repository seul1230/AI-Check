# mlp_strong_train.py
import os
import pandas as pd
import numpy as np
import joblib
import json
import datetime
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# from utils.extract_features import get_feature_names_v3
# from utils.extract_features_v4 import get_feature_names_v4
from utils.extract_features_v5 import get_feature_names_v5
from utils.extract_features_v6 import get_feature_names_v6
from utils.extract_features_v7 import get_feature_names_v7
from utils.extract_features_v8 import get_feature_names_v8
from utils.extract_features_v9 import get_feature_names_v9

# ✅ GPU 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
print("🖥️ Physical GPUs:", tf.config.list_physical_devices('GPU'))

# ✅ 설정
class_weight = "none"
model_name = "dacon-mlp-256-128-64-v9-auc-2-1"
# model_name = "mlp-256-128-64-v5-auc-2-1"
# model_name = "mlp-256-128-v4-recall"
learning_rate = 0.001
epoch = 100
batch_size = 32
SEED = 42

# ✅ 디렉토리 생성
today = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f"logs/{today}"
os.makedirs("logs", exist_ok=True)
os.makedirs("ckpt", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ✅ 데이터 불러오기
# df = pd.read_csv("dataset/final_url_dataset_with_features.csv").dropna()
# df = pd.read_csv("dataset/final_url_dataset_with_features_v4.csv").dropna()
# df = pd.read_csv("dataset/final_url_dataset_with_features_v5.csv").dropna()
df = pd.read_csv("dataset/train_dataset_with_features_v9.csv").dropna()
# df = pd.read_csv("dataset/final_url_dataset_with_features_v3.csv").dropna()

# ✅ 언더샘플링: label 1 수에 맞춰 label 0 샘플링
count_1 = sum(df["label"] == 1)
df_1 = df[df["label"] == 1]
# df_0 = df[df["label"] == 0].sample(n=round(count_1*1.5), random_state=SEED)
df_0 = df[df["label"] == 0].sample(n=round(count_1*2), random_state=SEED)

df_balanced = pd.concat([df_1, df_0]).sample(frac=1, random_state=SEED)

# X = df_balanced.drop(columns=["url", "domain", "label"])
# y = df_balanced["label"]

# selected_features = get_feature_names_v5()
selected_features = get_feature_names_v9()
X = df_balanced[selected_features]
y = df_balanced["label"]


# ✅ log1p 변환
log_transform_cols = [
    "url_len",
    "digit_count",
    "subdomain_count",
    "hostname_len"
]

# ✅ 변환 및 새로운 열로 추가
for col in selected_features:
    if col in df.columns:
        df[col] = np.log1p(df[col])

# ✅ train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ✅ y_test를 DataFrame으로 변환
y_test_df = pd.DataFrame(y_test, columns=["label"])
# ✅ X_test에 label 컬럼을 추가하여 저장
X_test_with_label = X_test.copy()
X_test_with_label["label"] = y_test_df.values

# ✅ CSV로 저장
X_test_with_label.to_csv("dataset/dacon_test_split_only_v9.csv", index=False)
print("✅ 테스트 데이터셋이 저장되었습니다: dataset/dacon_test_split_only_v9.csv")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_dict = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist()
}


joblib.dump(scaler, f"scaler/scaler_{model_name}.pkl")
with open(f"scaler/scaler_params_{model_name}.json", "w") as f:
    json.dump(scaler_dict, f)
    
print("🔍 NaN in X:", np.isnan(X_train_scaled).sum())
print("🔍 NaN in y:", np.isnan(y_train).sum())

# ✅ MLP 모델 정의
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    # Dropout(0.3),
    # Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

checkpoint_cb = ModelCheckpoint(
    filepath=f"ckpt/{model_name}_epoch_{epoch}_lr_{learning_rate}_batch_{batch_size}_{class_weight}_best_{today}.keras",
    save_best_only=True, monitor="val_auc", mode="max", verbose=1
)
early_stopping = EarlyStopping(monitor="val_auc", patience=5, restore_best_weights=True)
tensorboard_cb = TensorBoard(log_dir=log_dir)
lr_scheduler = ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=5, min_lr=1e-6)

# ✅ 학습
model.fit(
    X_train_scaled, y_train,
    epochs=epoch,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[checkpoint_cb, early_stopping, tensorboard_cb, lr_scheduler],
    verbose=1
)

# ✅ 평가
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)
print("✅ 평가 결과:\n", classification_report(y_test, y_pred))


# ✅ Accuracy Score 추가 출력
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 정확도 (Accuracy): {accuracy:.4f}")  # ✅ 추가 출력

# ✅ 저장
model.save(f"model/{model_name}_epoch_{epoch}_lr_{learning_rate}_batch_{batch_size}_{class_weight}_final_{today}.keras")
print(f"📌 model saved: model/{model_name}_epoch_{epoch}_lr_{learning_rate}_batch_{batch_size}_{class_weight}_final_{today}.keras")
