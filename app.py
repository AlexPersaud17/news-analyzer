from utils.classifier import classify_article
from utils.similarity import search_similar_articles
from utils.vector_store_init import setup_chromadb
from utils.sentiment import get_sentiment
from utils.vector_store_init import seed_db
import streamlit as st
import pandas as pd
from writerai import Writer
from dotenv import load_dotenv
import os
load_dotenv()

def llm(doc):
    api_key = os.getenv('WRITER_API_KEY')
    client = Writer(api_key=api_key)
    response = client.chat.chat(
        messages=[
            {"role": "system", "content": "You are an article title-creator. You will receive an input article text and you will give it an appropriate title."},
            {"role": "user", "content": doc}
        ],
        model="palmyra-x-004"
    )
    return response

def app():
    collection = setup_chromadb()
    # seed_db(collection)
    results = collection.get()
    docs = results["documents"]
    metadatas = results['metadatas']
    data = {
        "article": docs,
        "category": [metadata.get('category') for metadata in metadatas],
    }
    df = pd.DataFrame(data)
    st.write("### List of Articles")
    st.dataframe(df)
    # selected_article = st.radio("Choose an article to view", df['article'])
    # st.write("### Full Article")
    # st.write(selected_article)

if __name__ == "__main__":
    app()