import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.datasets import load_diabetes


diabetes = load_diabetes()

x = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.023, random_state=44)

# model = LinearRegression() 
model = Ridge(alpha=0.1) 
# model = Lasso(alpha=0.1)
# model = RandomForestRegressor(n_estimators=100,max_depth=4,min_samples_split=10,random_state=42) #60%

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(y_pred)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(mse)
print(r2)

mape = mean_absolute_percentage_error(y_test, y_pred)
accuracy = 100 - mape * 100
print(accuracy)
