import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load Iris dataset
df = pd.read_csv("Iris.csv")
df = df.drop('Id', axis=1, errors='ignore')

# Select 2 features and binary target
X = df[['SepalLengthCm', 'SepalWidthCm']].values
y = (df['Species'] == 'Iris-setosa').astype(int).values

svm = SVC(kernel="linear", C=1)
svm.fit(X, y)

# Predict & Evaluate
y_pred = svm.predict(X)
print("Accuracy:", round(accuracy_score(y, y_pred), 4))

# Get Decision Boundary and Margins
w = svm.coef_[0]
b = svm.intercept_[0]
x_values = np.array([X[:, 0].min(), X[:, 0].max()])

# w0*x + w1*y + b = 0
y_values = -(w[0] * x_values + b) / w[1]
# Upper margin: w0*x + w1*y + b = 1
y_upper = -(w[0] * x_values + b - 1) / w[1]
# Lower margin: w0*x + w1*y + b = -1
y_lower = -(w[0] * x_values + b + 1) / w[1]

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Setosa", color='blue')
plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Non-Setosa", color='orange')
plt.plot(x_values, y_values, label="Decision Boundary", color='black')
plt.plot(x_values, y_upper, "--", label="Upper Margin", color='grey')
plt.plot(x_values, y_lower, "--", label="Lower Margin", color='grey')

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Linear SVM Decision Boundary - Iris Dataset")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
