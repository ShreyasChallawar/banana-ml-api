# ===========================================================
#  Banana Shelf Life ML API (Final Deployment Version)
# ===========================================================

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from flask_cors import CORS

# -------------------------------------------
# Initialize Flask App
# -------------------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------------------
# File Paths (All must be in same folder)
# -------------------------------------------
MODEL_PATH = "banana_status_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

# -------------------------------------------
# Load Model Components
# -------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    print("✅ Model, Scaler, and Encoder loaded successfully.")
except Exception as e:
    print("❌ Error loading model components:", e)
    model = None
    scaler = None
    encoder = None

# -------------------------------------------
# Store Latest Reading for Debug / Frontend
# -------------------------------------------
latest = {
    "ethylene": None,
    "co2": None,
    "temperature": None,
    "humidity": None,
    "time_hrs": None,
    "predicted_status": None
}

# -------------------------------------------
# Health Check Endpoint
# -------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# -------------------------------------------
# Prediction Endpoint
# -------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None or encoder is None:
        return jsonify({"error": "Model or components not loaded"}), 500

    try:
        # Parse JSON Input
        data = request.get_json(force=True)

        eth = float(data.get("ethylene", 0))
        co2 = float(data.get("co2", 0))
        temp = float(data.get("temperature", 0))
        hum = float(data.get("humidity", 0))
        time_hrs = float(data.get("time_hrs", 0))

        # Arrange features in training order
        X = np.array([[eth, co2, temp, hum, time_hrs]])

        # Scale features
        X_scaled = scaler.transform(X)

        # Predict
        y_pred_encoded = model.predict(X_scaled)

        # Decode Label
        prediction = encoder.inverse_transform(y_pred_encoded)[0]

        # Save latest values
        latest.update({
            "ethylene": eth,
            "co2": co2,
            "temperature": temp,
            "humidity": hum,
            "time_hrs": time_hrs,
            "predicted_status": prediction
        })

        return jsonify({"predicted_status": prediction})

    except Exception as e:
        print("❌ Prediction Error:", e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# -------------------------------------------
# Retrieve Latest Values
# -------------------------------------------
@app.route("/latest", methods=["GET"])
def get_latest():
    return jsonify(latest)

# -------------------------------------------
# Main Entry Point
# -------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Flask API on port {port} ...")
    app.run(host="0.0.0.0", port=port)
