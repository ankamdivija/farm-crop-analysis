import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
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

model=tree.DecisionTreeClassifier(max_depth=11)
model.fit(X_train, y_train)

pickle.dump(model, open("decision_tree.pkl", "wb"))