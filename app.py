# app.py
from flask import Flask, jsonify
import joblib
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows cross-origin requests (important for MIT App Inventor)

# Load model
model = joblib.load("heatwave_model_.pkl")

# Weather API settings
API_KEY = "48e297f39f3b40b99b545750252606"
BASE_URL = "https://api.weatherapi.com/v1/current.json"
LOCATION = "Delhi"  # Change if needed

# Heat index formula
def compute_heat_index(T, H):
    return 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (H * 0.094))

@app.route('/')
def home():
    return "🔥 Heatwave Classifier API Running!"

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Fetch live weather from WeatherAPI
        url = f"{BASE_URL}?key={API_KEY}&q={LOCATION}"
        response = requests.get(url).json()

        if "current" not in response:
            return jsonify({"error": "Unable to fetch weather data"}), 500

        temp = response["current"]["temp_c"]
        humidity = response["current"]["humidity"]
        heatindex = compute_heat_index(temp, humidity)

        # Prediction
        prediction = model.predict([[temp, humidity, heatindex]])[0]
        label = "Heatwave likely" if prediction == 1 else "No heatwave"

        # Return all details
        return jsonify({
            "temp": temp,
            "humidity": humidity,
            "heatindex": heatindex,
            "prediction": label
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
