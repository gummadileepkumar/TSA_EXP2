### Developed by: Gumma Dileep Kumar
### Register No: 212222240032
### Date:

# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION

### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python On infy_stock dataset.

### ALGORITHM:
1. Import necessary libraries (NumPy, Matplotlib)
2. Load the dataset
3. Calculate the linear trend values using least square method
4. Calculate the polynomial trend values using least square method
5. End the program
### PROGRAM:
#### A - LINEAR TREND ESTIMATION
```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
data=pd.read_csv('infy_stock.csv')
data.head()
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data['Date'] = data['Date'].apply(lambda x: x.toordinal())
X = data['Date'].values.reshape(-1, 1)
y = data['Volume'].values
linear_model = LinearRegression()
linear_model.fit(X, y)
data['Linear_Trend'] = linear_model.predict(X)
plt.figure(figsize=(10,6))
plt.plot(data['Date'], data['Volume'],label='Original Data')
plt.plot(data['Date'], data['Linear_Trend'], color='orange', label='Linear Trend')
plt.title('Linear Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.grid(True)
plt.show()
```
#### B- POLYNOMIAL TREND ESTIMATION
```python
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
data[' Volume'] = poly_model.predict(X_poly)
plt.figure(figsize=(10,6))
plt.bar(data['Date'], data['Volume'], label='Original Data', alpha=0.6)
plt.plot(data['Date'], data[' Volume'],color='yellow', label='Poly Trend(Degree 2)')
plt.title('Polynomial Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.grid(True)
plt.show()
```
### Dataset:

## OUTPUT
A - LINEAR TREND ESTIMATION

![TSA_2 1](https://github.com/user-attachments/assets/93be642e-a2ea-4c1b-9ad3-24a68af439e0)


B- POLYNOMIAL TREND ESTIMATION
![TSA_2 2](https://github.com/user-attachments/assets/0b7e592f-b0dc-48e1-9d30-91da1e2f0661)



### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
