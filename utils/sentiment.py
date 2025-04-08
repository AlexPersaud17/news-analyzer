import joblib

def get_sentiment(new_article, classifier, vectorizer):
    classifier = joblib.load('models/saved_models/sentiment_model.pkl')
    vectorizer= joblib.load('models/saved_models/sentiment_vectorizer.pkl')
    new_article_vectorized = vectorizer.transform([new_article])
    sentiment = classifier.predict(new_article_vectorized)
    return sentiment