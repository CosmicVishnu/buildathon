import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")

X_train = train_df.drop('label', axis=1).values[:5000] / 255.0
y_train = train_df['label'].values[:5000]

X_test = test_df.drop('label', axis=1).values[:1000] / 255.0
y_test = test_df['label'].values[:1000]

learning_rates = [0.001, 0.01]
batch_sizes = [32, 64]
epochs = [20, 40]

results = []

for lr in learning_rates:
    for batch in batch_sizes:
        for epoch in epochs:
            model = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                learning_rate_init=lr,
                batch_size=batch,
                max_iter=epoch,
                random_state=42
            )
            start = time.time()
            model.fit(X_train, y_train)
            elapsed = time.time() - start

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            results.append([lr, batch, epoch, accuracy, elapsed])
            print(f"LR: {lr} | Batch: {batch} | Epochs: {epoch} -> Acc: {round(accuracy, 4)} ({round(elapsed, 2)}s)")

result_df = pd.DataFrame(
    results,
    columns=['Learning Rate', 'Batch Size', 'Epochs', 'Accuracy', 'Training Time']
)

print("\nHyperparameter Comparison:")
print(result_df)

best = result_df.loc[result_df['Accuracy'].idxmax()]
print("\nBest Hyperparameters:")
print(best)

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(results) + 1), result_df['Accuracy'], color='skyblue')
plt.xlabel("Config #")
plt.ylabel("Accuracy")
plt.title("Accuracy per Combination")

plt.subplot(1, 2, 2)
plt.bar(range(1, len(results) + 1), result_df['Training Time'], color='lightcoral')
plt.xlabel("Config #")
plt.ylabel("Time (s)")
plt.title("Training Time per Combination")

plt.tight_layout()
plt.show()
