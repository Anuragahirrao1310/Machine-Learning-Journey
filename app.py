from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # allows the frontend to talk to backend

model = joblib.load('model.pkl')  # load model once at startup

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()         # get data sent from frontend
    size = data['size']               # extract house size
    prediction = model.predict([[size]])[0]  # predict
    return jsonify({'price': round(prediction, 2)})  # send back

if __name__ == '__main__':
    app.run(debug=True)
    