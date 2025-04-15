import numpy as np
import pandas as pd
import joblib
from urllib.parse import urlparse
import re
import math
import tensorflow as tf

# from utils.extract_features_v4 import extract_lexical_features_v4, get_feature_names_v4
# from utils.extract_features_v5 import extract_lexical_features_v5, get_feature_names_v5
from utils.extract_features_v6 import extract_lexical_features_v6, get_feature_names_v6


# 🔁 입력 URL 예시
test_urls = [
    "http://secure-login.bankofamerica.cn/update",
    "https://www.naver.com",
    "www.naver.com",
    "westcoastveterinarycenterfl.com",
    "https://bit.ly/fake-login",
    "http://google.com/account/update/info?step=2"
]

# ✅ 특징 추출
# features = extract_lexical_features_v4(test_url)
# features = np.array(features).reshape(1, -1)


# ✅ log1p 변환
log_transform_cols = [
    "url_len", "digit_count", "subdomain_count", "hostname_len"
]

# # feature_names = get_feature_names_v4()
# # df_feat = pd.DataFrame(features, columns=feature_names)

# feature_list = [extract_lexical_features_v4(url) for url in test_urls]
# feature_names = get_feature_names_v4()
# df_feat = pd.DataFrame(feature_list, columns=feature_names)

# for col in log_transform_cols:
#     df_feat[col] = np.log1p(df_feat[col])

# ✅ 스케일러 로딩 및 변환
scaler = joblib.load("scaler/scaler_v6.pkl")
# scaled_input = scaler.transform(df_feat)

# ✅ TFLite 모델 로딩
interpreter = tf.lite.Interpreter(model_path="converted_model/mlp-256-128-64-v6-auc-2-1_epoch_50_lr_0.001_batch_32_none_final_2025-04-09_00-17-44.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# # ✅ 추론
# interpreter.set_tensor(input_details[0]['index'], scaled_input.astype(np.float32))
# interpreter.invoke()
# output = interpreter.get_tensor(output_details[0]['index'])
# print("📌 URL: ", test_urls) 
# print("✅ 추론 결과 (악성 확률):", output[0][0])


print("🔍 URL 추론 결과:")
for url in test_urls:
    feat = extract_lexical_features_v6(url)
    df = pd.DataFrame([feat], columns=get_feature_names_v6())

    for col in log_transform_cols:
        df[col] = np.log1p(df[col])

    scaled = scaler.transform(df).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], scaled)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0][0]

    print(f"- {url:60s} → 악성 확률: {output:.4f}")