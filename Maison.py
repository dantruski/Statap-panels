import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import ssl
import nltk
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from nltk.tokenize import word_tokenize

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt_tab')

# Chargement du dataset JSON
with open("temp_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def unify_discussion(item):
    text_parts = [f"{turn['role']}: {turn['content']}" for turn in item["history"]]
    text_parts.append("Question: " + item["question"])
    text_parts.append("Answer: " + item["answer"])
    return " ".join(text_parts)

docs = [unify_discussion(x) for x in data]

# Chargement de labels
df_labels = pd.read_excel("Topic_Data.xlsx").sort_values("Discussion")

# Ajustement des tailles 
min_len = min(len(df_labels), len(docs))
df_labels, docs = df_labels.iloc[:min_len], docs[:min_len]

# Préparation des documents pour Doc2Vec
tagged_docs = [TaggedDocument(words=word_tokenize(text.lower()), tags=[i]) for i, text in enumerate(docs)]

# ---- Création et entraînement du modèle ----
model_dbow = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, dm=0)
model_dbow.build_vocab(tagged_docs)
model_dbow.train(tagged_docs, total_examples=model_dbow.corpus_count, epochs=20)

# ---- Extraction des embeddings ----
doc_embeddings = np.array([model_dbow.dv[i] for i in range(len(docs))])

# ---- Alignement des labels ----
df_labels["index_python"] = df_labels["Discussion"] - 1
df_labels = df_labels.sort_values("index_python")
labels = df_labels["Label"].tolist()

# ---- Réduction de dimension avec PCA ----
pca = PCA(n_components=2)
X_pca = pca.fit_transform(doc_embeddings)

# ---- Visualisation ----
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["label"] = labels

unique_labels = df_pca["label"].unique()
cmap = cm.get_cmap("tab20", len(unique_labels))

plt.figure(figsize=(10, 6))
for i, lab in enumerate(unique_labels):
    subset = df_pca[df_pca["label"] == lab]
    plt.scatter(subset["PC1"], subset["PC2"], color=cmap(i), alpha=0.7, label=lab)

for i, row in df_pca.iterrows():
    plt.text(row["PC1"] + 0.01, row["PC2"] + 0.01, str(i), fontsize=7)

plt.title("Doc2Vec + PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc="best", fontsize=7)
plt.grid(True)
plt.show()

