# mlp_distill_train.py
import os
import pandas as pd
import numpy as np
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
print("🖥️ Physical GPUs:", tf.config.list_physical_devices('GPU'))

model_name = "randomforest_distill"

# ✅ 저장 디렉토리 생성
os.makedirs("logs", exist_ok=True)
os.makedirs("ckpt", exist_ok=True)
os.makedirs("model", exist_ok=True)

today = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f"logs/{today}"

# 1. 데이터 불러오기
df = pd.read_csv("dataset/final_url_dataset_with_features.csv")
X = df.drop(columns=["url", "domain", "label"])
y = df["label"]

# 2. 정규화
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 트리 모델 (teacher) 학습
teacher = RandomForestClassifier(n_estimators=200, random_state=42)
teacher.fit(X_train_scaled, y_train)
teacher_probs = teacher.predict_proba(X_train_scaled)

# 4. student MLP 정의
student = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])


student.compile(optimizer=Adam(0.001), loss="categorical_crossentropy", 
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

epoch = 30
batch_size = 128

# ✅ 콜백 설정
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"ckpt/{model_name}_epoch_{epoch}_batch_{batch_size}_best_{today}.keras",
    save_best_only=True, monitor="val_recall", mode="max", verbose=1
)
early_stopping = EarlyStopping(monitor="val_recall", patience=3, restore_best_weights=True)
tensorboard_cb = TensorBoard(log_dir=log_dir)
lr_scheduler = ReduceLROnPlateau(monitor="val_recall", factor=0.5, patience=5, min_lr=1e-6)


# 5. 학습
student.fit(X_train_scaled, teacher_probs, epochs=epoch, batch_size=batch_size,
            validation_split=0.25,  
            callbacks=[checkpoint_cb, early_stopping, tensorboard_cb, lr_scheduler],
            verbose=1)

# metrics_names = ['Loss', 'Accuracy', 'Recall', 'AUC']
# results = model.evaluate(test_ds, verbose=1)
# print(final_model_path)
# print("🎯 Final Evaluation Results:")
# for name, val in zip(metrics_names, results):
#     print(f"  - {name:<10}: {val:.4f}")

# 6. 평가
y_pred = student.predict(X_test_scaled)
y_pred_label = np.argmax(y_pred, axis=1)
print("✅ 평가 결과:\n", classification_report(y_test, y_pred_label))

# 7. 모델 저장
student.save(f"model/{model_name}_{today}.h5")
