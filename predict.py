import numpy as np
import json
from PIL import Image
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant_disease_prediction_EfficientNetB0_model.h5")
CLASS_INDICES_PATH = os.path.join(os.path.dirname(__file__), "class_indices.json")

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

def build_model(num_classes=38):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    base = EfficientNetB0(include_top=False, weights=None, input_tensor=inputs)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

print("Building model architecture...")
model = build_model()

print("Loading weights from .h5 file...")
model.load_weights(MODEL_PATH, by_name=False)
print("Model loaded successfully!")

with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    # EfficientNetB0 expects pixels in [0, 255] — it handles normalization internally
    return img_array


def predict_image(image_path):
    processed_img = load_and_preprocess_image(image_path)
    prediction = model.predict(processed_img, verbose=0)
    predicted_index = int(np.argmax(prediction))
    predicted_class = class_indices[str(predicted_index)]
    confidence = float(np.max(prediction))
    return predicted_class, confidence
