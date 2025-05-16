from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__)

model = pickle.load(open('model (1).pkl', 'rb'))

with open("beacon_mapping.json", "r") as f:
    beacon_mapping = json.load(f)

feature_cols = ['Beacon Code', 'Min Temp (°C)', 'Avg Temp (°C)', 'Max Temp (°C)', 'Precipitation (mm)', 'Max Wind (kph)', 'Avg Humidity (%)']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Map beacon name to encoded code
        beacon_name = data.get("Beacon")
        if beacon_name not in beacon_mapping:
            return jsonify({'error': f"Unknown Beacon: {beacon_name}. Please use a valid beacon name."})
        data["Beacon Code"] = beacon_mapping[beacon_name]
        data.pop("Beacon")  # Remove the original string value

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Predict
        prediction = model.predict(input_df[feature_cols])
        return jsonify({'predicted_settlement': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(debug=False, host="0.0.0.0", port=port)  # Bind to 0.0.0.0