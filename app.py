import pandas as pd
import numpy as np
import os
import signal
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
from pmdarima import auto_arima  # Auto-ARIMA for automatic order selection

# Ignore warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Enables CORS for frontend access

# Timeout handler to prevent long computations
def handler(signum, frame):
    raise Exception("Processing Timeout")

signal.signal(signal.SIGALRM, handler)

def forecast_stock(sales_data, periods=7):
    predictions = {}

    for item_name, sales_history in sales_data.items():
        try:
            if not isinstance(sales_history, list) or len(sales_history) < 3:
                predictions[item_name] = np.median(sales_history) if sales_history else 0  # Default prediction
                continue  

            # Ensure at least 5 data points by duplicating the last value
            while len(sales_history) < 5:
                sales_history.append(sales_history[-1])

            df = pd.DataFrame({"consumed_stock": sales_history})

            signal.alarm(10)  # Set timeout to 10 seconds per item

            # Auto-select best ARIMA order
            model = auto_arima(df['consumed_stock'], seasonal=False, suppress_warnings=True, stepwise=True)
            forecast = model.predict(n_periods=periods)

            signal.alarm(0)  # Reset timeout after success

            predicted_value = max(round(np.mean(forecast)), 0)  # Use average forecasted value
            predictions[item_name] = predicted_value  

        except Exception as e:
            predictions[item_name] = int(np.median(sales_history)) if sales_history else 0  # Fallback to median
            print(f"⚠️ Error processing {item_name}: {e}")  

    return predictions

@app.route('/')
def home():
    return "Stock Forecasting API is Running"

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
