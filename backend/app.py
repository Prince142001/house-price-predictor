from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load trained model with size check
model_path = 'house_price_model.pkl'
if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
    raise ValueError("Model file is missing or empty. Please upload a valid house_price_model.pkl.")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    area = data['area']
    bedrooms = data['bedrooms']
    bathrooms = data['bathrooms']

    prediction = model.predict(np.array([[area, bedrooms, bathrooms]]))
    price = round(prediction[0], 2)

    return jsonify({'predicted_price': f"${price}"})

if __name__ == '__main__':
    app.run(debug=True)
