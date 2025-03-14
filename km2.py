import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 1. Chargement des données
with open("temp_dataset.json", "r", encoding="utf-8") as file:
    data = json.load(file)

df = pd.DataFrame(data)

def extract_full_history(history):
    return " ".join([entry["role"] + ": " + entry["content"] 
                     for entry in history if "content" in entry]) if isinstance(history, list) else ""

df["history_full_text"] = df["history"].apply(extract_full_history)

# Fusion des colonnes textuelles
text_columns = ["history_full_text", "question", "query_modifier_question", "generated_answer", "answer"]
df_texts = df[text_columns].fillna("").apply(lambda x: " ".join(x), axis=1)

# 2. Vectorisation TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # Limite le nombre de dimensions
X_tfidf = vectorizer.fit_transform(df_texts.tolist())

# 3. Clustering K-Means
n_clusters = 5  # Ajustable
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(X_tfidf)
cluster_labels = kmeans.labels_

# 4. Réduction de dimension avec t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_tfidf.toarray())  # Transformer en array pour t-SNE

df_tsne = pd.DataFrame(X_tsne, columns=["Dim1", "Dim2"])
df_tsne["cluster"] = cluster_labels  # Ajout des clusters pour couleur

# 5. Visualisation
unique_clusters = df_tsne["cluster"].unique()
cmap = cm.get_cmap("tab10", n_clusters)

plt.figure(figsize=(12, 8))
for i, cluster_id in enumerate(unique_clusters):
    subset = df_tsne[df_tsne["cluster"] == cluster_id]
    plt.scatter(
        subset["Dim1"], 
        subset["Dim2"], 
        color=cmap(i),
        alpha=0.8,
        label=f"Cluster {cluster_id}"
    )

plt.title("Clustering K-Means sur TF-IDF (t-SNE en 2D)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend(loc="best")
plt.grid(True)
plt.show()
