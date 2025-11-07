# server.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---- Load model and preprocessors once at startup ----
MODEL_PATH = "banana_model.pkl"  # Must be in same folder as this file
loaded = joblib.load(MODEL_PATH)

# Extract stored objects
model = loaded["model"]
scaler = loaded.get("scaler")

# Try multiple possible key names for the label encoder
encoder = (
    loaded.get("le")
    or loaded.get("encoder")
    or loaded.get("label_encoder")
    or loaded.get("LabelEncoder")
)

# Keep latest reading for reference or frontend
latest = {
    "ethylene": None,
    "co2": None,
    "temperature": None,
    "humidity": None,
    "time_hrs": None,
    "predicted_status": None
}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON IN:
    {
      "ethylene": float,
      "co2": float,
      "temperature": float,
      "humidity": float,
      "time_hrs": float
    }

    JSON OUT:
    { "predicted_status": "ripe" }
    """
    data = request.get_json(force=True)

    try:
        eth = float(data.get("ethylene", 0))
        co2 = float(data.get("co2", 0))
        temp = float(data.get("temperature", 0))
        hum = float(data.get("humidity", 0))
        time_hrs = float(data.get("time_hrs", 0))
    except Exception as e:
        return jsonify({"error": f"Bad inputs: {e}"}), 400

    # Arrange in same order used during training
    X = np.array([[eth, co2, temp, hum, time_hrs]])

    # Apply scaling if scaler exists
    if scaler is not None:
        X = scaler.transform(X)

    try:
        # Predict encoded label
        y_pred_encoded = model.predict(X)

        # Decode prediction if encoder exists
        try:
            if encoder is not None:
                prediction = encoder.inverse_transform(y_pred_encoded)[0]
            else:
                prediction = str(y_pred_encoded[0])
        except Exception:
            # Fallback if inverse_transform fails
            prediction = str(y_pred_encoded[0])

    except Exception as e:
        return jsonify({"error": f"Model error: {e}"}), 500

    # Update latest reading
    latest.update({
        "ethylene": eth,
        "co2": co2,
        "temperature": temp,
        "humidity": hum,
        "time_hrs": time_hrs,
        "predicted_status": prediction
    })

    return jsonify({"predicted_status": prediction})


@app.route("/latest", methods=["GET"])
def get_latest():
    return jsonify(latest)


if __name__ == "__main__":
    # Locally run on port 5000; Render will inject PORT later
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
