# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 15:19:31 2020

@author: Kaushik
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os

app = Flask(__name__)
app.config['DEBUG'] = False
CORS(app)

port = int(os.getenv('PORT', 8090))

@app.route('/postreq', methods = ['POST'])
def covidCommSpreadPredReq():
    features = pd.read_csv('TrainingData/Dummy_Covid_Locality_Data.csv')    
    features = features.drop(['Locality'], axis=1)
    x = features.iloc[:, :-1].values
    y = features.iloc[:, -1].values
    le = LabelEncoder()
    y[:] = le.fit_transform(y[:])
    x[:, 1] = le.fit_transform(x[:, 1])
    x[:, 3] = le.fit_transform(x[:, 3])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=30, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    ann.fit(x_train, y_train, batch_size=32, epochs=100)
    
    
    if (request.is_json):
        reqdata = request.json
        content = pd.DataFrame([[
            reqdata['Positive_Adj_Locality'], 
            reqdata['People_travelling_through_infected_area'],
            reqdata['Aged_more_than_40'],
            reqdata['People_With_Symptom'],
            reqdata['Population']
            ]], 
            columns = [
                'Positive_Adj_Locality', 
                'People_travelling_through_infected_area', 
                'Aged_more_than_40', 
                'People_With_Symptom', 
                'Population'
        ])                
        req_data = content.iloc[:].values        
        req_data[:, 1] = le.fit_transform(req_data[:, 1])
        req_data[:, 3] = le.fit_transform(req_data[:, 3])
        req_data = sc.transform(req_data[:])
        req_data_pred = ann.predict(req_data)
        req_data_pred = req_data_pred > 0.5        
        return jsonify(str(req_data_pred[0][0]))
    else:
        return jsonify('Empty Request')
    
app.run(host='0.0.0.0', port= port)