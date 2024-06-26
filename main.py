import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the model
try:
    model = joblib.load('randomForestRegressor.pkl')
except Exception as e:
    print(f"Error loading the model: {e}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]  # Assuming model.predict expects a 2D array
        prediction = model.predict(final_features)
        return render_template('home.html', prediction_text=f"AQI for Jaipur: {prediction[0]}")
    except Exception as e:
        return render_template('home.html', prediction_text=f"Prediction error: {e}")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        prediction = model.predict([np.array(list(data.values()))])
        output = prediction[0]
        return jsonify(output)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
