import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from utils.preprocessing import preprocess_text
import pandas as pd
import joblib
import os

nltk.download('movie_reviews')
from nltk.corpus import movie_reviews

docs = [(movie_reviews.raw(fileid), category)
             for fileid in movie_reviews.fileids()
             for category in movie_reviews.categories(fileid)]

review_df = pd.DataFrame(docs, columns=["article", "label"])
review_df = preprocess_text(review_df)

X = review_df["cleaned_text"].tolist()
y = review_df["label"]

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
joblib.dump(tfidf_vectorizer, os.path.join(saved_models_dir, 'sentiment_vectorizer.pkl'))
joblib.dump(classifier, os.path.join(saved_models_dir, 'sentiment_model.pkl'))