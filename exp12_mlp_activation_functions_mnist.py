import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load MNIST dataset
df = pd.read_csv("mnist_784.csv")
X = df.drop('class', axis=1).values
y = df['class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Subset and normalize
X_train = (X_train[:10000]) / 255.0
y_train = y_train[:10000]
X_test = (X_test[:2000]) / 255.0
y_test = y_test[:2000]

activations = {
    'Sigmoid': 'logistic',
    'ReLU': 'relu',
    'Tanh': 'tanh'
}

accuracies = []
training_times = []
iterations = []

for name, activation in activations.items():
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation=activation,
        solver='adam',
        max_iter=50,
        random_state=42
    )
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    accuracies.append(acc)
    training_times.append(elapsed)
    iterations.append(model.n_iter_)

    print(f"\nActivation   : {name}")
    print("Accuracy     :", round(acc, 4))
    print("Training Time:", round(elapsed, 2), "s")
    print("Iterations   :", model.n_iter_)

# Comparison Plots
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].bar(activations.keys(), accuracies, color='dodgerblue')
axes[0].set_title("Accuracy")
axes[0].set_ylim(0, 1)

axes[1].bar(activations.keys(), training_times, color='salmon')
axes[1].set_title("Training Time (s)")

axes[2].bar(activations.keys(), iterations, color='mediumpurple')
axes[2].set_title("Iterations to Converge")

plt.tight_layout()
plt.show()
