from flask import Flask, jsonify, render_template, request, redirect
import pickle

import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

feature_names = ["Nitrogen","Phosphorous","Potassium","temperature","humidity","rainfall","ph","ph_type"]

prediction_labels = [
    {"id": 0, "crop": "Apple", "crop_image": "images/apple.jpg", "info": ["Moderate", "High", "Moderate", "21-24°C", "Moderate", "800-1000 mm", "6.0-6.5", "Slightly acidic"]},
    {"id": 1, "crop": "Banana", "crop_image": "images/banana.jpg", "info": ["High", "Moderate", "High", "26-30°C", "High", "1000-1200 mm", "5.5-6.5", "Slightly acidic"]},
    {"id": 2, "crop": "blackgram", "crop_image": "images/blackgram.jpg", "info": ["Low", "Moderate", "Low", "25-35°C", "Moderate to high", "600-1000 mm", "6.5-7.5", "Neutral to slightly alkaline"]},
    {"id": 3, "crop": "chickpea", "crop_image": "images/chickpea.jpg", "info": ["Low", "Moderate", "Low", "24-28°C", "Low to moderate", "600-1000 mm", "5.5-7.0", "Neutral to slightly acidic"]},
    {"id": 4, "crop": "coconut", "crop_image": "images/coconut.jpg", "info": ["Moderate", "High", "Moderate", "28-30°C", "High", "1000-1500 mm", "5.5-6.5", "Slightly acidic"]},
    {"id": 5, "crop": "coffee", "crop_image": "images/coffee.jpg", "info": ["High", "Moderate", "High", "18-25°C", "High", "1500-2500 mm", "6.0-6.5", "Slightly acidic"]},
    {"id": 6, "crop": "cotton", "crop_image": "images/cotton.jpg", "info": ["High", "Moderate", "High", "21-30°C", "Moderate", "500-700 mm", "5.5-7.5", "Neutral to slightly acidic"]},
    {"id": 7, "crop": "grapes", "crop_image": "images/grapes.jpg", "info": ["Moderate", "Moderate", "High", "15-30°C", "Low to moderate", "600-800 mm", "5.5-6.5", "Slightly acidic"]},
    {"id": 8, "crop": "jute", "crop_image": "images/jute.jpg", "info": ["High", "Low", "Low", "24-37°C", "High", "1000-1500 mm", "6.0-7.0", "Neutral"]},
    {"id": 9, "crop": "kidneybeans", "crop_image": "images/kidneybeans.jpg", "info": ["Moderate", "High", "Moderate", "18-23°C", "Moderate", "400-800 mm", "6.0-6.8", "Slightly acidic"]},
    {"id": 10, "crop": "lentil", "crop_image": "images/lentil.jpg", "info": ["Low", "Moderate", "Low", "18-24°C", "Moderate", "350-600 mm", "6.0-7.0", "Neutral"]},
    {"id": 11, "crop": "maize", "crop_image": "images/maize.jpg", "info": ["High", "Moderate", "High", "18-27°C", "Moderate to high", "500-800 mm", "5.5-7.0", "Neutral"]},
    {"id": 12, "crop": "mango", "crop_image": "images/mango.jpg", "info": ["Moderate", "Low", "Moderate", "24-27°C", "Moderate", "700-1000 mm", "5.5-7.5", "Neutral to slightly acidic"]},
    {"id": 13, "crop": "mothbeans", "crop_image": "images/mothbeans.jpg", "info": ["Low", "Moderate", "Low", "25-40°C", "Low", "200-600 mm", "5.0-7.0", "Highly variable"]},
    {"id": 14, "crop": "mungbean", "crop_image": "images/mungbean.jpg", "info": ["Low", "Moderate", "Low", "25-35°C", "Moderate", "300-500 mm", "6.0-7.2", "Neutral"]},
    {"id": 15, "crop": "muskmelon", "crop_image": "images/muskmelon.jpg", "info": ["High", "Moderate", "High", "21-28°C", "Low", "400-600 mm", "6.0-6.7", "Slightly acidic"]},
    {"id": 16, "crop": "orange", "crop_image": "images/orange.jpg", "info": ["Moderate", "High", "Moderate", "15-29°C", "Moderate to high", "900-1200 mm", "6.0-7.5", "Neutral to slightly acidic"]},
    {"id": 17, "crop": "papaya", "crop_image": "images/papaya.jpg", "info": ["High", "Moderate", "Moderate", "21-33°C", "High", "1200-2000 mm", "5.5-6.7", "Slightly acidic"]},
    {"id": 18, "crop": "pigeonpeas", "crop_image": "images/pigeonpeas.jpg", "info": ["Low", "Moderate", "Low", "20-30°C", "Moderate", "450-500 mm", "5.0-7.0", "Highly variable"]},
    {"id": 19, "crop": "pomegranate", "crop_image": "images/pomegranate.jpg", "info": ["Moderate", "High", "Moderate", "20-30°C", "Moderate", "600-800 mm", "5.5-7.0", "Neutral to slightly acidic"]},
    {"id": 20, "crop": "rice", "crop_image": "images/rice.jpg", "info": ["High", "High", "High", "20-30°C", "High", "1000-2500 mm", "5.0-6.5", "Acidic to neutral"]},
    {"id": 21, "crop": "watermelon", "crop_image": "images/watermelon.jpg", "info": ["Moderate", "Low", "Moderate", "22-28°C", "Low to moderate", "400-600 mm", "5.5-6.5", "Slightly acidic"]}
]


@app.route('/')
def hello() :
    return 'Hey!'

@app.route('/home')
def homepage() :
    return render_template("index.html")

@app.route('/knowmore')
def knowmore() :
    return render_template("knowmore.html")

@app.route('/predict-crop', methods=['POST'])
def predict_crop() :
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return prediction_labels[prediction[0]]


@app.route('/predict')
def predict() :
    return render_template("predict.html", feature_names=feature_names)

if __name__ == '__main__':
    app.run()