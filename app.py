import pandas as pd
import numpy as np
import os
import signal
from flask import Flask, request, jsonify
from statsmodels.tsa.arima.model import ARIMA
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enables CORS for frontend access

# Timeout handler to prevent Render timeouts
def handler(signum, frame):
    raise Exception("Processing Timeout")

signal.signal(signal.SIGALRM, handler)

def forecast_stock(sales_data, periods=7):
    predictions = {}

    for item_name, sales_history in sales_data.items():
        try:
            if not isinstance(sales_history, list) or len(sales_history) < 3:
                predictions[item_name] = 0  # Default prediction for bad data
                continue  

            # Ensure at least 5 data points
            while len(sales_history) < 5:
                sales_history.append(0)

            df = pd.DataFrame({"consumed_stock": sales_history})

            signal.alarm(10)  # Set timeout to 10 seconds per item

            # Simpler ARIMA model (lightweight for Render Free Plan)
            model = ARIMA(df['consumed_stock'], order=(1,1,0))  
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=periods)

            signal.alarm(0)  # Reset timeout after success

            predicted_value = max(round(forecast[-1]), 0)  # Ensure non-negative values
            predictions[item_name] = predicted_value  

        except Exception as e:
            predictions[item_name] = 0  # Default on failure
            print(f"⚠️ Error processing {item_name}: {e}")  

    return predictions

@app.route('/predict', methods=['POST'])
def predict():
    try:
        request_data = request.json
        sales_data = request_data.get("sales_data", {})

        if not sales_data:
            return jsonify({"error": "No sales data provided"}), 400

        predictions = forecast_stock(sales_data)
        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return 500 if something goes wrong

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render uses dynamic ports
    app.run(debug=False, host="0.0.0.0", port=port)  # Turn off debug for production
