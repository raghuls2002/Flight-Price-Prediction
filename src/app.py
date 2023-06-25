# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:53:02 2023

@author: user
"""

from preprocessing import cols, label_map, scaler    
import numpy as np

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('form.html', cols = cols, categorical_cols = list(label_map.keys()), label_map = label_map)

@app.route('/submit', methods=['POST'])
def register():
    args ={}
    
    for i in cols:
        if i in list(label_map.keys()):
            args[i]= request.form[i].strip()
            args[i] = label_map[i][args[i]]
        else:
            args[i]=float(request.form[i].strip())
                 
        
    with open("./model.pickle","rb") as file:
            mp = pickle.load(file)
            
    predicted_value = mp.predict(scaler.transform(np.array(list(args.values())).reshape(1, -1))) 
    print("Recommended career is "+ str(predicted_value[0]))
            
    return render_template('success.html', value = predicted_value)

if __name__ == '__main__':
    app.run(debug=True)
