# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:53:02 2023

@author: user
"""

from datetime import date, datetime, timedelta
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

label_map = {'airline': {'AirAsia': 0, 'Air_India': 1, 'GO_FIRST': 2, 'Indigo': 3, 'SpiceJet': 4, 'Vistara': 5}, 
             'source_city': {'Bangalore': 0, 'Chennai': 1, 'Delhi': 2, 'Hyderabad': 3, 'Kolkata': 4, 'Mumbai': 5}, 
             'stops': {'one': 0, 'two_or_more': 1, 'zero': 2},  
             'destination_city': {'Bangalore': 0, 'Chennai': 1, 'Delhi': 2, 'Hyderabad': 3, 'Kolkata': 4, 'Mumbai': 5}, 
             'class': {'Business': 0, 'Economy': 1}}

def time_categorizer(hr:int)->str:
    if hr>=0 and hr<=3:
        return 3
    elif hr >=4 and hr<=7 :
        return 1
    elif hr >=8 and hr<=11 :
        return 4
    elif hr >= 12 and hr<=17:
        return 0
    elif hr >= 18 and hr<=21:
        return 2
    elif hr >= 21 and hr<=24:
         return 5
    else:
         return -1
    
    
@app.route('/')
def home():
    return render_template('form.html',  label_map = label_map)

@app.route('/submit', methods=['POST'])
def register():
    airline = label_map["airline"][request.form["airline"]]    
    source_city = label_map["source_city"][request.form["source_city"]]    
    
    departure_datetime = datetime.strptime(request.form["departure_time"], '%Y-%m-%dT%H:%M')
    departure_time = time_categorizer(departure_datetime.hour) 
    
    stops = label_map["stops"][request.form["stops"]]    

    arrival_time_ = datetime.strptime(request.form["arrival_time"], '%H:%M').time()
    arrival_time = time_categorizer(arrival_time_.hour)
    
    destination_city = label_map["destination_city"][request.form["destination_city"]]    
    class_ = label_map["class"][request.form["class"]]    
    
    if departure_datetime.hour> arrival_time_.hour :
        arrival_datetime = datetime.combine(departure_datetime.date()+ timedelta(days=1), arrival_time_)
    else:
        arrival_datetime = datetime.combine(departure_datetime.date(), arrival_time_)
        
    time_diff = arrival_datetime - departure_datetime
    duration = time_diff.total_seconds() / 3600
    
    current_date = datetime.now()
    days_left = (departure_datetime - current_date).days

    with open("./model.pickle","rb") as file:
            mp = pickle.load(file)
            
    prediction = mp.predict([[
        airline, 
        source_city,
        departure_time,
        stops,
        arrival_time,
        destination_city,
        class_,
        duration,
        days_left
        ]])
            
    return render_template('success.html', text = str(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
