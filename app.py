import pandas as pd
import numpy as np
import os
import signal
from flask import Flask, request, jsonify
from statsmodels.tsa.arima.model import ARIMA
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Timeout handler
def handler(signum, frame):
    raise Exception("Processing Timeout")

signal.signal(signal.SIGALRM, handler)

def forecast_stock(sales_data, periods=7):  # Reduce forecast to 7 days
    predictions = {}

    for item_name, sales_history in sales_data.items():
        try:
            if not isinstance(sales_history, list) or len(sales_history) < 3:
                predictions[item_name] = 0  # Default prediction for bad data
                continue  

            while len(sales_history) < 5:
                sales_history.append(0)

            df = pd.DataFrame({"consumed_stock": sales_history})

            signal.alarm(20)  # Increase timeout to 20 seconds per item

            # Train a simpler ARIMA model
            model = ARIMA(df['consumed_stock'], order=(2,1,0))  # Reduced complexity
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=periods)

            signal.alarm(0)  # Reset timeout after success

            predicted_value = round(forecast[-1])  
            predictions[item_name] = max(predicted_value, 0)  

        except Exception as e:
            predictions[item_name] = 0  
            print(f"Error processing {item_name}: {e}")  

    return predictions

@app.route('/predict', methods=['POST'])
def predict():
    request_data = request.json
    sales_data = request_data.get("sales_data", {})

    if not sales_data:
        return jsonify({"error": "No sales data provided"}), 400

    predictions = forecast_stock(sales_data)
    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
