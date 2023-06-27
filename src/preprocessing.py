# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 17:26:22 2023

@author: user
"""

import pandas as pd 

# Loading dataset
data = pd.read_csv("../data/dataset.csv")
data = data.drop(['Unnamed: 0','flight'], axis=1)

target = 'price'
categorical_cols= data.select_dtypes(include=['object']).columns.tolist()

# Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

label_map = {}

for column in categorical_cols:
    le.fit(data[column])
    data[column] = le.transform(data[column])
    
    label_map[column] = dict(zip(le.classes_, le.transform(le.classes_)))
    
print(label_map)

# Splitting into X and y
y = data[target]
X = data.drop(columns=[target], axis = 1)


# Splitting into training and testing data
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# Scaling independent variables
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)




