import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np


def input_fitur(input_pnn50,input_rmssd):
    pred = mlp.predict([[input_pnn50,input_rmssd]])
    pred_prob = mlp.predict_proba([[input_pnn50,input_rmssd]])[0][0]
    return pred, pred_prob

    
dataset = pd.read_csv('Dataset.csv')
dataset.drop(['dataset','Note'],axis=1,inplace=True)
feature_set = dataset.drop(['label','SDNN','HR'],axis=1,inplace=False)
target = pd.DataFrame(dataset['label'])
x_train, x_test, y_train, y_test= train_test_split(feature_set, target, test_size=0.3)
mlp = MLPClassifier(hidden_layer_sizes=(2), activation='logistic', solver='lbfgs', max_iter=10000)
mlp.fit(x_train,y_train.values.ravel())
pred=mlp.predict(x_test)
akurasi=mlp.score(x_test,y_test)*100

