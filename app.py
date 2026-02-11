from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp = float(request.form['temperature'])
    hum = float(request.form['humidity'])
    wind = float(request.form['wind'])
    rain = float(request.form['rain'])

    features = np.array([[temp, hum, wind, rain]])
    result = model.predict(features)
    prediction = "🔥 Forest Fire Risk" if result[0] == 1 else "✅ No Fire Risk"

    return render_template('index.html', prediction_text=prediction)

if __name__ == '__main__':
    app.run(debug=True)
