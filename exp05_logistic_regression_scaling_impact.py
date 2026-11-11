import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load dataset
data = pd.read_csv("pima_indians_diabetes.csv")

# 2. Separate features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# WITHOUT FEATURE SCALING
# --------------------------------------------------
model_without_scaling = LogisticRegression(max_iter=1000)
model_without_scaling.fit(X_train, y_train)
y_pred_without = model_without_scaling.predict(X_test)

print("WITHOUT FEATURE SCALING")
print("Accuracy :", round(accuracy_score(y_test, y_pred_without), 4))
print("Precision:", round(precision_score(y_test, y_pred_without), 4))
print("Recall   :", round(recall_score(y_test, y_pred_without), 4))
print("F1-score :", round(f1_score(y_test, y_pred_without), 4))

# --------------------------------------------------
# WITH FEATURE SCALING
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_with_scaling = LogisticRegression(max_iter=1000)
model_with_scaling.fit(X_train_scaled, y_train)
y_pred_with = model_with_scaling.predict(X_test_scaled)

print("\nWITH FEATURE SCALING")
print("Accuracy :", round(accuracy_score(y_test, y_pred_with), 4))
print("Precision:", round(precision_score(y_test, y_pred_with), 4))
print("Recall   :", round(recall_score(y_test, y_pred_with), 4))
print("F1-score :", round(f1_score(y_test, y_pred_with), 4))
