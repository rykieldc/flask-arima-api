import pandas as pd
import numpy as np
import os
from flask import Flask, request, jsonify
from statsmodels.tsa.arima.model import ARIMA
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enables cross-origin requests

# Function to perform ARIMA forecasting for multiple items
def forecast_stock(sales_data, periods=30):
    predictions = {}

    for item_name, sales_history in sales_data.items():
        try:
            if not isinstance(sales_history, list):
                predictions[item_name] = 0  # Default prediction if input is invalid
                continue  

            # Ensure at least 5 data points by adding zeros if needed
            while len(sales_history) < 5:
                sales_history.append(0)

            # Convert sales history to DataFrame
            df = pd.DataFrame({"consumed_stock": sales_history})

            # Train ARIMA model
            model = ARIMA(df['consumed_stock'], order=(5,1,0))  # Adjust order if needed
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=periods)

            # Convert forecast to whole numbers
            predicted_value = round(forecast[-1])  # Taking the last predicted value
            predictions[item_name] = max(predicted_value, 0)  # Ensure no negative values

        except Exception as e:
            predictions[item_name] = 0  # Default prediction in case of errors

    return predictions

# API Endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.json
    sales_data = request_data.get("sales_data", {})

    if not sales_data:
        return jsonify({"error": "No sales data provided"}), 400

    predictions = forecast_stock(sales_data)
    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get PORT from environment
    app.run(debug=True, host="0.0.0.0", port=port)
