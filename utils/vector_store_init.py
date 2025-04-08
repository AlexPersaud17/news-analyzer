import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from chromadb.utils import embedding_functions
from sklearn.datasets import fetch_20newsgroups
from utils.preprocessing import preprocess_text
import pandas as pd


def setup_chromadb():
    client = chromadb.PersistentClient()
    collection = client.get_or_create_collection("news_articles_v2", embedding_function = DefaultEmbeddingFunction())
    return collection

def seed_db(collection):
    news_dataset = fetch_20newsgroups(subset='all')
    X = news_dataset.data
    y = news_dataset.target
    category_names = news_dataset.target_names

    X_df = pd.DataFrame(X, columns=["article"])
    X_df['category'] = [category_names[id] for id in y]
    X_df = preprocess_text(X_df)

    # 1280
    for idx, row in X_df.iterrows():
        print(idx)
        collection.add(
            documents=[row['article']],
            metadatas=[{'category': row['category']}],
            ids=[f"article_{idx}"]
            )