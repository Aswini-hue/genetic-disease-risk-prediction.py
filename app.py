

from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model

import os

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "genetic_risk_api_model.joblib"
)

saved_data = joblib.load(MODEL_PATH)
model = saved_data["model"]
input_columns = saved_data["input_columns"]

print("Loaded model from:", MODEL_PATH)
print("Model type:", type(model).__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Genetic Disease Risk Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "JSON input is required"}), 400

        # Accept one patient record as a JSON object
        input_df = pd.DataFrame([data])

        # Validate required input features
        missing = [
            col for col in input_columns
            if col not in input_df.columns
        ]

        if missing:
            return jsonify({
                "error": "Missing required input features",
                "missing_columns": missing
            }), 400

        input_df = input_df[input_columns]

        prediction = model.predict(input_df)[0]

        return jsonify({
            "predicted_disease_risk": str(prediction),
            "note": "Educational project prediction; not a medical diagnosis."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
