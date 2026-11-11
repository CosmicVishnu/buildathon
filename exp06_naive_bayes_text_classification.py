import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score

# 1. Load datasets
train_data = pd.read_csv("20news_train.csv")
test_data = pd.read_csv("20news_test.csv")

# 2. Separate text and labels
X_train_text = train_data['text'].fillna('')
y_train = train_data['label']

X_test_text = test_data['text'].fillna('')
y_test = test_data['label']

# --------------------------------------------------
# 3. MULTINOMIAL NAIVE BAYES
# --------------------------------------------------
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)

print("MULTINOMIAL NAIVE BAYES")
print("Accuracy:", round(accuracy_score(y_test, y_pred_mnb), 4))
print("F1-score:", round(f1_score(y_test, y_pred_mnb, average='weighted'), 4))

# --------------------------------------------------
# 4. BERNOULLI NAIVE BAYES
# --------------------------------------------------
binary_vectorizer = CountVectorizer(binary=True)
X_train_binary = binary_vectorizer.fit_transform(X_train_text)
X_test_binary = binary_vectorizer.transform(X_test_text)

bnb = BernoulliNB()
bnb.fit(X_train_binary, y_train)
y_pred_bnb = bnb.predict(X_test_binary)

print("\nBERNOULLI NAIVE BAYES")
print("Accuracy:", round(accuracy_score(y_test, y_pred_bnb), 4))
print("F1-score:", round(f1_score(y_test, y_pred_bnb, average='weighted'), 4))
