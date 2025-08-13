# --- 데이터셋 다운로드 ---

# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("vinayakshanawad/meat-freshness-image-dataset")

# print("Path to dataset files:", path)


import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomContrast,
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ----- 변수 -----
IMAGE_SIZE = 224
BATCH_SIZE = 32
TRAIN_DIR = r"Meat Freshness.v1-new-dataset.multiclass/train/"
VALID_DIR = r"Meat Freshness.v1-new-dataset.multiclass/valid/"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "meat_fresh_classifier.h5")

# ----- 데이터 로드 -----
train_df = pd.read_csv(os.path.join(TRAIN_DIR, "_classes.csv"))
valid_df = pd.read_csv(os.path.join(VALID_DIR, "_classes.csv"))

train_df["filename"] = train_df["filename"].str.strip()
valid_df["filename"] = valid_df["filename"].str.strip()
train_image_paths = TRAIN_DIR + train_df["filename"]
valid_image_paths = VALID_DIR + valid_df["filename"]
train_labels = train_df[[" Fresh", " Half-Fresh", " Spoiled"]].values
valid_labels = valid_df[[" Fresh", " Half-Fresh", " Spoiled"]].values
class_names = ["Fresh", "Half-Fresh", "Spoiled"]

train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
validation_ds = tf.data.Dataset.from_tensor_slices((valid_image_paths, valid_labels))


def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label


# ----- 데이터 증강: 조명, 대비 변화 -----
def random_brightness_fn(image, label):
    image = tf.image.random_brightness(image, max_delta=0.2)  # 밝기 변화
    return image, label


data_augmentation = tf.keras.Sequential(
    [
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
        RandomZoom(0.2),
        RandomContrast(0.2),
    ]
)

AUTOTUNE = tf.data.AUTOTUNE


# --- 전처리 함수 ---
def prepare(ds, shuffle=False, augment=False):
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(BATCH_SIZE)
    if augment:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
        ds = ds.map(random_brightness_fn, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(buffer_size=AUTOTUNE)


train_ds = prepare(train_ds, shuffle=True, augment=True)
validation_ds = prepare(validation_ds)

# ----- 모델 구성 -----
base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)
base_model.trainable = False

inputs = base_model.input
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(3, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# ----- 학습 -----
history = model.fit(
    train_ds,
    epochs=10,
    validation_data=validation_ds,
    callbacks=[tensorboard_callback, early_stop],
)

# ----- 파인튜닝 -----
base_model.trainable = True
for layer in base_model.layers[:-20]:  # 앞부분 동결
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_ds,
    epochs=20,
    validation_data=validation_ds,
    callbacks=[tensorboard_callback, early_stop],
)

# ----- 모델 저장 -----
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_PATH)

# ----- 학습 결과 시각화 -----
acc = history.history["accuracy"] + history_ft.history["accuracy"]
val_acc = history.history["val_accuracy"] + history_ft.history["val_accuracy"]
loss = history.history["loss"] + history_ft.history["loss"]
val_loss = history.history["val_loss"] + history_ft.history["val_loss"]

# 학습 결과 그래프 저장
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.ylabel("Accuracy")
plt.ylim([min(plt.ylim()), 1])
plt.title("Training and Validation Accuracy")

plt.subplot(2, 1, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.ylabel("Cross Entropy")
plt.ylim([0, 1.0])
plt.title("Training and Validation Loss")
plt.xlabel("epoch")

os.makedirs("plots", exist_ok=True)
plt.savefig("plots/training_validation_metrics.png", dpi=300, bbox_inches="tight")
plt.close()

# ----- 상세 평가 지표 -----
y_pred_list = []
y_true_list = []
for image_batch, label_batch in validation_ds:
    pred = model.predict(image_batch, verbose=0)
    y_pred_list.extend(np.argmax(pred, axis=1))
    y_true_list.extend(np.argmax(label_batch, axis=1))

print("\n--- Classification Report ---")
print(classification_report(y_true_list, y_pred_list, target_names=class_names))

# 혼동 행렬 저장
conf_matrix = confusion_matrix(y_true_list, y_pred_list)
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.savefig("plots/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
