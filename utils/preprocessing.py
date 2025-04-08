import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pandas as pd

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def preprocess_text(data):
    data["article_lowercase"] = data["article"].str.lower()
    en_stopwords = set(stopwords.words("english"))
    data["article_no_stopwords"] = data["article_lowercase"].apply(
        lambda x: " ".join(
            [word for word in x.split() if word not in (en_stopwords)]
        )
    )
    data["articles_no_stopwords_no_punct"] = data["article_no_stopwords"].apply(
        lambda x: re.sub(r"[^\w\s]", "", x)
    )
    data["tokenized"] = data["articles_no_stopwords_no_punct"].apply(
        lambda x: word_tokenize(x)
    )
    lemmatizer = WordNetLemmatizer()
    data["lemmatized"] = data["tokenized"].apply(
        lambda tokens: [lemmatizer.lemmatize(token) for token in tokens]
    )
    data["cleaned_text"] = data["lemmatized"].apply(
        lambda tokens: " ".join(tokens)
    )
    return data
