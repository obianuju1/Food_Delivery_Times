import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
regmodel = pickle.load(open('best_model.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))  # Load the scaler if applicable

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Prediction API
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    return jsonify(output[0])

# Prediction form route
@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    prediction = regmodel.predict(final_input)[0]
    return render_template('home.html', prediction_text=f"Predicted Delivery Time: {prediction:.2f} minutes")

if __name__ == "__main__":
    app.run(debug=True)
