from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import gdown

BASE_DIR = os.path.dirname(__file__)

MODELS = [
    {
        "name": "EfficientNetB0",
        "path": os.path.join(BASE_DIR, "plant_disease_prediction_EfficientNetB0_model.h5"),
        "drive_id": "1WvwfHEcFq9JU4iBlOqxYz57voqfLN4I3"
    },
    {
        "name": "InceptionV3",
        "path": os.path.join(BASE_DIR, "InceptionV3.h5"),
        "drive_id": "1_NDUXHnQAzPSRoiRzK5jbwGnsZcauh7k"
    },
    {
        "name": "ResNet50",
        "path": os.path.join(BASE_DIR, "ResNet50.keras"),
        "drive_id": "11agLpoI55du-knEPvv416kWEkLCRupcl"
    },
    {
        "name": "VGG16",
        "path": os.path.join(BASE_DIR, "VGG16.h5"),
        "drive_id": "1p6k8DvbwTqllg7VRAEJ9ZwpHW3i_nxJO"
    },
]

for m in MODELS:
    if not os.path.exists(m["path"]):
        print(f"Downloading {m['name']} from Google Drive...")
        gdown.download(
            f"https://drive.google.com/uc?id={m['drive_id']}",
            m["path"], quiet=False, fuzzy=True
        )
        print(f"{m['name']} downloaded successfully!")

from predict import predict_image_ensemble

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    import uuid
    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        predicted_class, confidence, votes = predict_image_ensemble(filepath)
        return jsonify({
            "disease": predicted_class,
            "confidence": round(float(confidence), 4),
            "votes": votes
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/")
def home():
    return jsonify({"message": "KrishiRakshak ML service running ✅", "version": "2.0.0"})


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
