import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Load California Housing dataset
data = pd.read_csv("california_housing.csv")
data = data.dropna()[:1500]

X = data[['AveRooms']].values
y = data['MedHouseVal'].values

# Standardization
mean = X.mean()
std = X.std()
X_norm = ((X - mean) / std).flatten()

# Gradient Descent
b0 = 0
b1 = 0
lr = 0.01
ini = 1000
n = len(X_norm)

for i in range(ini):
    yp = b0 + b1 * X_norm
    db0 = (2 / n) * np.sum(yp - y)
    db1 = (2 / n) * np.sum(X_norm * (yp - y))
    b0 = b0 - db0 * lr
    b1 = b1 - db1 * lr

yg = b0 + b1 * X_norm

print("Gradient Descent MSE:", mean_squared_error(y, yg))
print("Gradient Descent R2 :", r2_score(y, yg))

# Normal Equation
Xn = np.column_stack((np.ones(len(X_norm)), X_norm))
t = np.linalg.inv(Xn.T @ Xn) @ Xn.T @ y
b0n = t[0]
b1n = t[1]

yn = b0n + b1n * X_norm

print("Normal Equation MSE :", mean_squared_error(y, yn))
print("Normal Equation R2  :", r2_score(y, yn))

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Actual Data points", alpha=0.5)
plt.plot(X, yg, label="GD", color="red")
plt.plot(X, yn, '--', label="NE", color="green")
plt.xlabel("AveRooms")
plt.ylabel("MedHouseVal")
plt.title("Linear Regression: Gradient Descent vs Normal Equation")
plt.legend()
plt.tight_layout()
plt.show()
