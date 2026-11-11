import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")

# Separate features and labels
X_train = train_df.drop('label', axis=1).values[:5000] / 255.0
y_train = train_df['label'].values[:5000]

X_test = test_df.drop('label', axis=1).values[:1000] / 255.0
y_test = test_df['label'].values[:1000]

# Define SVM kernels
models = {
    'Linear': SVC(kernel='linear', C=1),
    'Polynomial': SVC(kernel='poly', degree=3, C=1, gamma='scale'),
    'RBF': SVC(kernel='rbf', C=1, gamma='scale')
}

accuracies = []
training_times = []

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    elapsed_time = time.time() - start

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    accuracies.append(accuracy)
    training_times.append(elapsed_time)

    print(f"\n{name} Kernel:")
    print("Accuracy     :", round(accuracy, 4))
    print("Training Time:", round(elapsed_time, 2), "seconds")

# Visualizations
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(models.keys(), accuracies, color='steelblue')
plt.xlabel("SVM Kernel")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.ylim(0, 1)

plt.subplot(1, 2, 2)
plt.bar(models.keys(), training_times, color='coral')
plt.xlabel("SVM Kernel")
plt.ylabel("Training Time (s)")
plt.title("Training Time Comparison")

plt.tight_layout()
plt.show()
