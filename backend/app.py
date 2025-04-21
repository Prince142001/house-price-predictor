# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model
with open('house_price_model.pkl', 'rb') as f:
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
