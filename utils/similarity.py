import joblib

def query_similarity(article, collection):
    results = collection.query([article], n_results=5)
    return results

def search_similar_articles(new_article, collection, predicted_category):
    vectorizer = joblib.load('models/saved_models/vectorizer.pkl')
    results = query_similarity(new_article, vectorizer, collection)
    similar_articles = []
    for i, result in enumerate(results['documents']):
        if results['metadatas'][i]["category"] == predicted_category:
            similar_articles.append(result)
    return similar_articles
