import joblib

def classify_article(new_article, classifier, vectorizer):
    classifier = joblib.load('models/saved_models/classifier.pkl')
    vectorizer = joblib.load('models/saved_models/vectorizer.pkl')
    new_article_vectorized = vectorizer.transform([new_article])
    predicted_category = classifier.predict(new_article_vectorized)
    return predicted_category