import numpy as np
import json
import os
from PIL import Image
import tensorflow as tf

BASE_DIR = os.path.dirname(__file__)
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "class_indices.json")

with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

NUM_CLASSES = len(class_indices)


print("Loading all models...")

efficientnet = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "plant_disease_prediction_EfficientNetB0_model.h5"), compile=False)
print("EfficientNetB0 loaded ✅")

inceptionv3 = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "InceptionV3.h5"), compile=False)
print("InceptionV3 loaded ✅")

resnet50 = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "ResNet50.keras"), compile=False)
print("ResNet50 loaded ✅")

vgg16 = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "VGG16.h5"), compile=False)
print("VGG16 loaded ✅")

MODELS = [
    {"name": "EfficientNetB0", "model": efficientnet},
    {"name": "InceptionV3",    "model": inceptionv3},
    {"name": "ResNet50",       "model": resnet50},
    {"name": "VGG16",          "model": vgg16},
]


def preprocess(image_path, target_size):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict_image_ensemble(image_path):
    vote_counts = {}
    confidences = {}

    for m in MODELS:
        # Read input size directly from the model
        _, h, w, _ = m["model"].input_shape
        img = preprocess(image_path, (h, w))
        preds = m["model"].predict(img, verbose=0)
        idx = int(np.argmax(preds))
        cls = class_indices[str(idx)]
        conf = float(np.max(preds))

        vote_counts[cls] = vote_counts.get(cls, 0) + 1
        confidences.setdefault(cls, []).append(conf)

    # Majority vote winner
    max_votes = max(vote_counts.values())
    top_classes = [c for c, v in vote_counts.items() if v == max_votes]

    # Tie-break by average confidence
    winner = max(top_classes, key=lambda c: np.mean(confidences[c]))
    avg_confidence = float(np.mean(confidences[winner]))

    votes_summary = {cls: count for cls, count in vote_counts.items()}

    return winner, avg_confidence, votes_summary
