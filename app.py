from statistics import StatisticsError, mode
from flask import Flask, jsonify, render_template, request, redirect
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
import logging

app = Flask(__name__)

app.logger.setLevel(logging.DEBUG)  # Set log level to INFO
handler = logging.FileHandler('app.log')  # Log to a file
app.logger.addHandler(handler)

models = [pickle.load(open("knn.pkl", "rb")), pickle.load(open("logistic.pkl", "rb")), pickle.load(open("naive_bayes.pkl", "rb")), pickle.load(open("random_forest.pkl", "rb")),
          pickle.load(open("svc.pkl", "rb")), pickle.load(open("xgb.pkl", "rb")), pickle.load(open("decision_tree.pkl", "rb"))]

feature_names = ["Nitrogen","Phosphorous","Potassium","temperature","humidity","rainfall","ph","ph_type"]

prediction_labels = [
    {"id": 0, "crop": "Apple", "crop_image": "images/apple.jpeg", "info": ["Moderate", "High", "Moderate", "21-24°C", "Moderate", "800-1000 mm", "6.0-6.5", "Slightly acidic"]},
    {"id": 1, "crop": "Banana", "crop_image": "images/banana.jpeg", "info": ["High", "Moderate", "High", "26-30°C", "High", "1000-1200 mm", "5.5-6.5", "Slightly acidic"]},
    {"id": 2, "crop": "Blackgram", "crop_image": "images/blackgram.jpeg", "info": ["Low", "Moderate", "Low", "25-35°C", "Moderate to high", "600-1000 mm", "6.5-7.5", "Neutral to slightly alkaline"]},
    {"id": 3, "crop": "Chickpea", "crop_image": "images/chickpea.jpeg", "info": ["Low", "Moderate", "Low", "24-28°C", "Low to moderate", "600-1000 mm", "5.5-7.0", "Neutral to slightly acidic"]},
    {"id": 4, "crop": "Coconut", "crop_image": "images/coconut.jpeg", "info": ["Moderate", "High", "Moderate", "28-30°C", "High", "1000-1500 mm", "5.5-6.5", "Slightly acidic"]},
    {"id": 5, "crop": "Coffee", "crop_image": "images/coffee.jpeg", "info": ["High", "Moderate", "High", "18-25°C", "High", "1500-2500 mm", "6.0-6.5", "Slightly acidic"]},
    {"id": 6, "crop": "Cotton", "crop_image": "images/cotton.jpeg", "info": ["High", "Moderate", "High", "21-30°C", "Moderate", "500-700 mm", "5.5-7.5", "Neutral to slightly acidic"]},
    {"id": 7, "crop": "Grapes", "crop_image": "images/grapes.jpeg", "info": ["Moderate", "Moderate", "High", "15-30°C", "Low to moderate", "600-800 mm", "5.5-6.5", "Slightly acidic"]},
    {"id": 8, "crop": "Jute", "crop_image": "images/jute.jpeg", "info": ["High", "Low", "Low", "24-37°C", "High", "1000-1500 mm", "6.0-7.0", "Neutral"]},
    {"id": 9, "crop": "Kidneybeans", "crop_image": "images/kidneybeans.jpeg", "info": ["Moderate", "High", "Moderate", "18-23°C", "Moderate", "400-800 mm", "6.0-6.8", "Slightly acidic"]},
    {"id": 10, "crop": "Lentil", "crop_image": "images/lentil.jpg", "info": ["Low", "Moderate", "Low", "18-24°C", "Moderate", "350-600 mm", "6.0-7.0", "Neutral"]},
    {"id": 11, "crop": "Maize", "crop_image": "images/maize.jpeg", "info": ["High", "Moderate", "High", "18-27°C", "Moderate to high", "500-800 mm", "5.5-7.0", "Neutral"]},
    {"id": 12, "crop": "Mango", "crop_image": "images/mango.jpeg", "info": ["Moderate", "Low", "Moderate", "24-27°C", "Moderate", "700-1000 mm", "5.5-7.5", "Neutral to slightly acidic"]},
    {"id": 13, "crop": "Mothbeans", "crop_image": "images/mothbeans.jpeg", "info": ["Low", "Moderate", "Low", "25-40°C", "Low", "200-600 mm", "5.0-7.0", "Highly variable"]},
    {"id": 14, "crop": "Mungbean", "crop_image": "images/mungbean.jpeg", "info": ["Low", "Moderate", "Low", "25-35°C", "Moderate", "300-500 mm", "6.0-7.2", "Neutral"]},
    {"id": 15, "crop": "Muskmelon", "crop_image": "images/muskmelon.jpeg", "info": ["High", "Moderate", "High", "21-28°C", "Low", "400-600 mm", "6.0-6.7", "Slightly acidic"]},
    {"id": 16, "crop": "Orange", "crop_image": "images/orange.jpeg", "info": ["Moderate", "High", "Moderate", "15-29°C", "Moderate to high", "900-1200 mm", "6.0-7.5", "Neutral to slightly acidic"]},
    {"id": 17, "crop": "Papaya", "crop_image": "images/papaya.jpeg", "info": ["High", "Moderate", "Moderate", "21-33°C", "High", "1200-2000 mm", "5.5-6.7", "Slightly acidic"]},
    {"id": 18, "crop": "Pigeonpeas", "crop_image": "images/pigeonpeas.jpeg", "info": ["Low", "Moderate", "Low", "20-30°C", "Moderate", "450-500 mm", "5.0-7.0", "Highly variable"]},
    {"id": 19, "crop": "Pomegranate", "crop_image": "images/pomegranate.jpeg", "info": ["Moderate", "High", "Moderate", "20-30°C", "Moderate", "600-800 mm", "5.5-7.0", "Neutral to slightly acidic"]},
    {"id": 20, "crop": "Rice", "crop_image": "images/rice.jpg", "info": ["High", "High", "High", "20-30°C", "High", "1000-2500 mm", "5.0-6.5", "Acidic to neutral"]},
    {"id": 21, "crop": "Watermelon", "crop_image": "images/watermelon.jpeg", "info": ["Moderate", "Low", "Moderate", "22-28°C", "Low to moderate", "400-600 mm", "5.5-6.5", "Slightly acidic"]}
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


@app.route('/analyse')
def analyse() :
    crop_df=pd.read_csv("Crop_Data_Cleaned.csv")

    crop_df.drop('crop_label',axis=1,inplace=True)
    crop_label_encoder = LabelEncoder()
    crop_df['crop_label'] = crop_label_encoder.fit_transform(crop_df['label'])

    crop_df.drop(['ph_type','label'],axis=1,inplace=True)
    df = crop_df[crop_df["humidity"] >= 80]

    humidity_data = [prediction_labels[x] for x in df["crop_label"].unique().tolist()]

    df = crop_df[crop_df["rainfall"] >= 200]

    rainfall_data = [prediction_labels[x] for x in df["crop_label"].unique().tolist()]

    df = crop_df[crop_df["ph"] >= 7.5]

    ph_data = [prediction_labels[x] for x in df["crop_label"].unique().tolist()]


    return render_template("analyse.html", humidity_data=humidity_data, ph_data=ph_data, rainfall_data=rainfall_data)

def predict_with_majority_vote(models, features_array):
    predictions = [model.predict(features_array)[0] for model in models]
    
    try:
        final_prediction = mode(predictions)
    except StatisticsError:
        # Handle case where there is no unique mode by selecting the first one
        final_prediction = predictions[0]
        print("No unique mode, selected the first prediction.")
    
    return final_prediction

@app.route('/predict-crop', methods=['POST'])
def predict_crop() :
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]

    return prediction_labels[predict_with_majority_vote(models, features)]


@app.route('/predict')
def predict() :
    return render_template("predict.html", feature_names=feature_names)

if __name__ == '__main__':
    app.run(debug=True)