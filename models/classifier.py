import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import fetch_20newsgroups
from utils.preprocessing import preprocess_text
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

news_dataset = fetch_20newsgroups(subset="all")
X = news_dataset.data
y = news_dataset.target

X_df = pd.DataFrame(X, columns=["article"])
X_df = preprocess_text(X_df)
X = X_df["cleaned_text"].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)
y_pred = classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

report = classification_report(y_test, y_pred)
print(report)

saved_models_dir = 'models/saved_models/'
joblib.dump(tfidf_vectorizer, os.path.join(saved_models_dir, 'vectorizer.pkl'))
joblib.dump(classifier, os.path.join(saved_models_dir, 'classifier.pkl'))