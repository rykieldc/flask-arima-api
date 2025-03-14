import pandas as pd
import numpy as np
import os
import signal
from flask import Flask, request, jsonify
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima  # Auto ARIMA to find best order
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Timeout handler
def handler(signum, frame):
    raise Exception("Processing Timeout")

signal.signal(signal.SIGALRM, handler)

def forecast_stock(sales_data, periods=7):
    predictions = {}

    for item_name, sales_history in sales_data.items():
        try:
            if not isinstance(sales_history, list) or len(sales_history) < 5:
                predictions[item_name] = 0  # Default for bad data
                continue  

            df = pd.DataFrame({"consumed_stock": sales_history})

            signal.alarm(20)  # 20s timeout per item

            # Find best ARIMA order
            auto_model = auto_arima(df['consumed_stock'], seasonal=False, suppress_warnings=True, stepwise=True)

            # Train ARIMA with best order
            best_order = auto_model.order
            model = ARIMA(df['consumed_stock'], order=best_order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=periods)

            signal.alarm(0)  # Reset timeout

            # Take the last forecasted value, round it, and ensure it's non-negative
            predicted_value = max(round(forecast[-1]), 0)
            predictions[item_name] = predicted_value  

        except Exception as e:
            predictions[item_name] = 0  
            print(f"⚠️ Error processing {item_name}: {e}")  # Log the error

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
