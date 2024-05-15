# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:16:04 2024

@author: aspdi
"""

from flask import Flask, request
import json
from tensorflow.keras import models
import joblib

smoker_encoder=joblib.load(r"smoker_encoder (1).pkl")
region_encoder=joblib.load(r"region_en.pkl")
gen_encoder=joblib.load(r"gen_encoder (6).pkl")

model=models.load_model(r"insurance (1).h5")
import pandas as pd
app=Flask(__name__)
@app.route('/',methods=['POST'])

def testing():
    data=request.get_json(force=True)
    print(data)
    data=pd.DataFrame([data])
    print(data)
    print(type(data))
    data['sex']=gen_encoder.transform(data['sex'])
    data['smoker']=smoker_encoder.transform(data['smoker'])
    print(data)
    region_data=region_encoder.transform(data[['region']]).toarray()
    print(region_data)
    region_data1=pd.DataFrame(region_data,columns=['region_northeast', 'region_northwest', 'region_southeast',
       'region_southwest'])
    print(region_data1)
    finaldata=pd.concat([data,region_data1],axis=1)
    print(finaldata)
    finaldata=finaldata.drop('region',axis='columns')
    output=model.predict(finaldata)
    
        
    return str(output)
    
