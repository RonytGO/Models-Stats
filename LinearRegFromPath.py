import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from Excel file
file_path = "C:\\Users\\Ronyt\\OneDrive - Shalom Hartman Institute\\Desktop\\Example.xlsx"  # Replace with your actual file path
df = pd.read_excel(file_path)

# Assuming the Excel file has columns 'X' and 'Y'
x = df['X'].values
y = df['Y'].values

# Compute linear regression coefficients
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]

# Generate regression line
y_pred = m * x + c

# Plot data points
plt.scatter(x, y, color='red', label='Data Points')
plt.plot(x, y_pred, color='blue', label=f'Linear Fit: y = {m:.2f}x + {c:.2f}')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.title('Linear Regression Example')
plt.grid()
plt.show()
