import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load Digits CSV dataset
df = pd.read_csv("digits.csv")
X = df.drop('label', axis=1).values
y = df['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_values = [5, 8, 10, 12, 15]
inertias = []
silhouette_scores = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_
    score = silhouette_score(X_scaled, labels)

    inertias.append(inertia)
    silhouette_scores.append(score)

    print(f"Clusters: {k:2d} | Inertia: {round(inertia, 2):10.2f} | Silhouette Score: {round(score, 4)}")

# Visualizations
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(k_values, inertias, marker='o', color='purple')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o', color='teal')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores")
plt.grid(True)
plt.tight_layout()
plt.show()
