# export_to_onnx.py
import tensorflow as tf
import tf2onnx
from tensorflow.keras.applications import EfficientNetB0

IMAGE_SIZE = 224
H5_PATH = "model/meat_fresh_classifier.h5"
ONNX_PATH = "model/meat_fresh_classifier.onnx"


def build_classifier():
    base = EfficientNetB0(
        include_top=False, weights=None, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(3, activation="softmax")(x)
    return tf.keras.Model(inputs=base.input, outputs=out)


model = build_classifier()
model.load_weights(H5_PATH)  # H5에서 가중치만 로드

# 동적 배치 입력 시그니처
spec = (tf.TensorSpec((None, IMAGE_SIZE, IMAGE_SIZE, 3), tf.float32, name="input"),)

# ONNX로 변환
onnx_model, _ = tf2onnx.convert.from_keras(
    model, input_signature=spec, opset=13, output_path=ONNX_PATH
)
print(f"Saved: {ONNX_PATH}")
