import pandas as pd
from sklearn.datasets import load_diabetes  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

diabetes = load_diabetes()

newdf = pd.DataFrame(diabetes.data, columns= diabetes.feature_names)
newdf['target'] = diabetes.target
print(newdf.head())

x = newdf.drop('target', axis=1)
y = newdf['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=47) 

model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

new1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(new1.head())

print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  
plt.grid(True)
plt.show()
