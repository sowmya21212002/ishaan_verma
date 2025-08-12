# app.py
from flask import Flask, jsonify
import joblib
import requests

app = Flask(__name__)

# Load trained model
model = joblib.load("heatwave_model.pkl")

# Weather API settings
API_KEY = "48e297f39f3b40b99b545750252606"
BASE_URL = "https://api.weatherapi.com/v1/current.json"
LOCATION = "Delhi"

def compute_heat_index(T, H):
    return 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (H * 0.094))

@app.route("/predict", methods=["GET"])
def predict():
    # Fetch live weather
    url = f"{BASE_URL}?key={API_KEY}&q={LOCATION}"
    response = requests.get(url).json()

    if "current" not in response:
        return jsonify({"error": "Unable to fetch weather data"}), 500

    temp = response["current"]["temp_c"]
    humidity = response["current"]["humidity"]
    heatindex = compute_heat_index(temp, humidity)

    # Make prediction
    pred = model.predict([[temp, humidity, heatindex]])[0]
    result = "YES" if pred == 1 else "NO"

    return jsonify({
        "temp": temp,
        "humidity": humidity,
        "heatindex": heatindex,
        "prediction": result
    })

if __name__ == "__main__":
    app.run()
