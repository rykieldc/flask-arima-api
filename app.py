import pandas as pd
import numpy as np
import os
from flask import Flask, request, jsonify
from statsmodels.tsa.arima.model import ARIMA
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows cross-origin requests

# Function to perform ARIMA forecasting
def forecast_stock(data, periods=30):
    try:
        df = pd.DataFrame(data)  # Convert input to DataFrame
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Ensure column exists
        if 'consumed_stock' not in df.columns:
            return {"error": "Missing 'consumed_stock' column"}

        # Train ARIMA model
        model = ARIMA(df['consumed_stock'], order=(5,1,0))  # Adjust order if needed
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=periods)

        return {"forecast": forecast.tolist()}
    
    except Exception as e:
        return {"error": str(e)}

# API Endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.json
    data = request_data.get("history", [])

    if not data:
        return jsonify({"error": "No data provided"}), 400

    predictions = forecast_stock(data)
    return jsonify(predictions)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get PORT from environment
    app.run(debug=True, host="0.0.0.0", port=port)
