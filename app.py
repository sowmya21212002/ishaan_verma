# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows cross-origin requests (important for MIT App Inventor)

# Load model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "🔥 Heatwave Classifier API Running!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            data = request.get_json(force=True)
            temp = float(data.get('temp'))
            humidity = float(data.get('humidity'))
            heatindex = float(data.get('heatindex'))
        else:  # GET method
            temp = float(request.args.get('temp'))
            humidity = float(request.args.get('humidity'))
            heatindex = float(request.args.get('heatindex'))

        # Prediction
        prediction = model.predict([[temp, humidity, heatindex]])[0]
        label = "Heatwave likely" if prediction == 1 else "No heatwave"
        
        return jsonify({'prediction': label})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)

