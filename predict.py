import numpy as np
import json
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0, InceptionV3, ResNet50, VGG16

BASE_DIR = os.path.dirname(__file__)
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "class_indices.json")

with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

NUM_CLASSES = len(class_indices)


def build_efficientnet(num_classes):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    base = EfficientNetB0(include_top=False, weights=None, input_tensor=inputs)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)


def build_inceptionv3(num_classes):
    inputs = tf.keras.Input(shape=(299, 299, 3))
    base = InceptionV3(include_top=False, weights=None, input_tensor=inputs)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)


def build_resnet50(num_classes):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    base = ResNet50(include_top=False, weights=None, input_tensor=inputs)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)


def build_vgg16(num_classes):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    base = VGG16(include_top=False, weights=None, input_tensor=inputs)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)


print("Loading all models...")

efficientnet = build_efficientnet(NUM_CLASSES)
efficientnet.load_weights(os.path.join(BASE_DIR, "plant_disease_prediction_EfficientNetB0_model.h5"), by_name=False)
print("EfficientNetB0 loaded ✅")

inceptionv3 = build_inceptionv3(NUM_CLASSES)
inceptionv3.load_weights(os.path.join(BASE_DIR, "InceptionV3.h5"), by_name=False)
print("InceptionV3 loaded ✅")

resnet50 = tf.keras.models.load_model(os.path.join(BASE_DIR, "ResNet50.keras"))
print("ResNet50 loaded ✅")

vgg16 = build_vgg16(NUM_CLASSES)
vgg16.load_weights(os.path.join(BASE_DIR, "VGG16.h5"), by_name=False)
print("VGG16 loaded ✅")

MODELS = [
    {"name": "EfficientNetB0", "model": efficientnet, "size": (224, 224)},
    {"name": "InceptionV3",    "model": inceptionv3,  "size": (299, 299)},
    {"name": "ResNet50",       "model": resnet50,      "size": (224, 224)},
    {"name": "VGG16",          "model": vgg16,         "size": (224, 224)},
]


def preprocess(image_path, target_size):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict_image_ensemble(image_path):
    vote_counts = {}   # class_name -> vote count
    confidences = {}   # class_name -> list of confidence scores

    for m in MODELS:
        img = preprocess(image_path, m["size"])
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

    # Build votes summary for response
    votes_summary = {cls: count for cls, count in vote_counts.items()}

    return winner, avg_confidence, votes_summary
