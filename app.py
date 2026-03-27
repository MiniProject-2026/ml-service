from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Auto-download model from Google Drive if not present
MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant_disease_prediction_EfficientNetB0_model.h5")
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    import gdown
    gdown.download(
        f"https://drive.google.com/uc?id=1WvwfHEcFq9JU4iBlOqxYz57voqfLN4I3",
        MODEL_PATH, quiet=False
    )
    print("Model downloaded successfully!")

from predict import predict_image

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

    # Use a safe unique filename
    import uuid
    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        predicted_class, confidence = predict_image(filepath)
        return jsonify({
            "disease": predicted_class,
            "confidence": round(float(confidence), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/")
def home():
    return jsonify({"message": "KrishiRakshak ML service running ✅", "version": "1.0.0"})


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
