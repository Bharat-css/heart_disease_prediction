from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load your saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Home route (optional HTML form)
@app.route('/')
def home():
    return render_template('index.html')  # Create an index.html for form input

# Prediction route (API style)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)

        if features.shape[1] != 14:
            return render_template('index.html', prediction=None, error="Expected 14 features.")

        prediction = model.predict(features)
        return render_template('index.html', prediction=int(prediction[0]))
    except Exception as e:
        return render_template('index.html', prediction=None, error=str(e))
    
if __name__ == '__main__':
    app.run(debug=True)