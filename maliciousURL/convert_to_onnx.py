import joblib
import numpy as np
import onnx
from skl2onnx import convert_sklearn
# from skl2onnx.common.data_types import FloatTensorType

from xgboost import XGBClassifier
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from skl2onnx.common._registration import register_converter, register_shape_calculator
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from skl2onnx import update_registered_converter


import os

# ✅ 경로 설정
# file_name = "tfidf_1000_xgb_2000_2025_04_01_07_11_17_acc_9336"
file_name = "tfidf_3000_xgb_1000_2025_04_07_07_13_55_acc_9336"
model_path = f"model/{file_name}.pkl"
scaler_path = f"scaler/tf_idf_boost.pkl"
onnx_output_path = f"converted_model/{file_name}_xgboost.onnx"

# register_converter('XGBClassifier', convert_xgboost)
# register_shape_calculator('XGBClassifier', calculate_linear_classifier_output_shapes)

# update_registered_converter(
#     XGBClassifier,
#     'XGBoostXGBClassifier',
#     calculate_linear_classifier_output_shapes,
#     convert_xgboost,
#     options={'zipmap': [False], 'predict_proba': [True]}
# )


# ✅ 모델 및 스케일러 로드
model = joblib.load(model_path)

# # ✅ Booster 로드
# model = xgb.Booster()
# model.load_model(model_path)  # ← JSON 불러오는 방식



# ✅ ONNX 변환
# 예: 입력 shape이 17차원이라고 하면
# scaler = joblib.load(scaler_path)
input_dim = 3021
# input_dim = 1021
initial_type = [('input', FloatTensorType([None, input_dim]))]

# onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=13)
# onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=13)
onnx_model = convert_xgboost(
    model,
    initial_types=initial_type,
    target_opset=13
)

# ✅ 저장
with open(onnx_output_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
onnx.checker.check_model(onnx.load(onnx_output_path))
print(f"✅ ONNX 변환 완료: {onnx_output_path}")

model = onnx.load(onnx_output_path)
print("Inputs:")
print(model.graph.input)
print("Outputs:")
print(model.graph.output)
