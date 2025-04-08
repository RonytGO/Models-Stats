# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:18:37 2025

@author: Ronyt
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# נתונים
X = np.array([93, 95, 80, 91, 88, 76]).reshape(-1, 1)
Y = np.array([88, 82, 78, 86, 92, 65])

# Creating the Linear Regression Model
model = LinearRegression()
model.fit(X, Y)

# חיזוי ערכים
X_pred = np.linspace(min(X), max(X), 100).reshape(-1, 1)
Y_pred = model.predict(X_pred)

# הצגת התוצאה
plt.scatter(X, Y, color='blue', label="original data")
plt.plot(X_pred, Y_pred, color='red', label=f'regression line\nY={model.coef_[0]:.2f}X+{model.intercept_:.2f}')
plt.xlabel("Average Score in the bagrut (X)")
plt.ylabel("Average Score in the Toar (Y)")
plt.legend()
plt.grid()
plt.show()

# הצגת המשוואה
print(f"משוואת הקו: Y = {model.coef_[0]:.2f}X + {model.intercept_:.2f}")
