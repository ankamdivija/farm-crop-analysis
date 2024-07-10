import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB 
import pickle
from sklearn.model_selection import train_test_split

crop_df=pd.read_csv("Crop_Data_Cleaned.csv")

crop_df.drop('crop_label',axis=1,inplace=True)
crop_label_encoder = LabelEncoder()
crop_df['crop_label'] = crop_label_encoder.fit_transform(crop_df['label'])

crop_df.drop(['ph_type','label'],axis=1,inplace=True)

X=crop_df.drop('crop_label',axis=1)
y=crop_df['crop_label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train,y_train)

pickle.dump(nb_model, open("naive_bayes.pkl", "wb"))