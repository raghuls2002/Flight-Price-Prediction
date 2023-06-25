# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 23:08:43 2023

@author: user
"""

from preprocessing import X_train, X_test, y_train, y_test

# Building ML model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)


# Training ML Model
model.fit(X_train, y_train)


# Testing ML Model
y_pred = model.predict(X_test)
print(y_pred)


# Performance Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE): ", mse)
print("Mean Absolute Error (MAE): ", mae)
print("R-squared (R2) Score: ", r2)


# Saving ML Model
import pickle
with open('model.pickle', 'wb') as file:
   pickle.dump(model, file)