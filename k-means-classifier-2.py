##Embedding avec un K-means

#Importation des librairies
import os
import random
import re
import string

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk import word_tokenize
from nltk.corpus import stopwords


from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download("stopwords")
nltk.download("punkt")

SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)

corpus = [
    "The king rules the kingdom with strength.",
    "The king is a powerful and courageous man.",
    "The queen leads the palace with wisdom.",
    "The queen is an elegant and intelligent woman.",
    "The man works hard and stands by his ideas.",
    "The woman brings gentleness and perseverance.",
    "In the past, kings and queens were respected.",
    "The king loves war, but peace reigns thanks to the queen.",
    "The king and queen govern the land together.",
    "The man admires the wisdom of the queen.",
    "The woman respects the strength of the king.",
    "The kingdom thrives under the rule of the king and queen.",
    "The queen values peace and harmony in the palace.",
    "The man seeks courage and strength like the king.",
    "The woman aspires to wisdom and elegance like the queen.",
    "The king's bravery inspires the people of the kingdom.",
    "The queen's intelligence guides the decisions of the court.",
    "The man and woman work together for the prosperity of the kingdom."
]
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[{}]".format(string.punctuation), " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    return " ".join(tokens)

corpus_adapte=[clean_text(text) for text in corpus]
corpus_adapte 

class EmbeddingKMeans:
    def __init__(self, n_clusters=5, max_iter=1000, n_init=100, tol=1e-3):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.vectorizer = TfidfVectorizer()
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            n_init=self.n_init,
            tol=self.tol,
            random_state=SEED
        )

    def fit(self, corpus):
        X = self.vectorizer.fit_transform(corpus)
        self.kmeans.fit(X)
        self.embeddings = X
        return self

    def get_embeddings(self):
        return self.embeddings


embedding_model = text_embedding_kmeans.vectorizer
embedding_model.fit(corpus)
embedding_king = embedding_model.transform(['king']).toarray()
embedding_queen = embedding_model.transform(['queen']).toarray()
embedding_man = embedding_model.transform(['man']).toarray()
embedding_woman = embedding_model.transform(['woman']).toarray()

print("Embedding pour 'reine algébrique':", embedding_king - embedding_man + embedding_woman)
print("Embedding pour 'reine':", embedding_queen)
print(embedding_king - embedding_man + embedding_woman == embedding_queen)



#Trouver le nombre optimal de clusters en mesurant le score de silhouette
def find_optimal_clusters(corpus, max_clusters=7):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    silhouette_scores = []

    for n_clusters in range(2, max_clusters + 1):
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED)
        kmeans.fit(X)
        labels = kmeans.labels_
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title("Silhouette Score vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid()
    plt.show()


find_optimal_clusters(corpus_adapte)
#n=5 est un nombre optimal de clusters