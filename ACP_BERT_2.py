import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Chargement des données
with open("temp_dataset.json", "r", encoding="utf-8") as file:
    data = json.load(file)

df = pd.DataFrame(data)

def extract_full_history(history):
    return " ".join([entry["role"] + ": " + entry["content"] for entry in history if "content" in entry]) if isinstance(history, list) else ""

df["history_full_text"] = df["history"].apply(extract_full_history)

# Fusion des colonnes textuelles
text_columns = ["history_full_text", "question", "query_modifier_question", "generated_answer", "answer"]
df_texts = df[text_columns].fillna("").apply(lambda x: " ".join(x), axis=1)

# Chargement des labels
df_labels = pd.read_excel("Topic_Data.xlsx")
labels = df_labels["Label"].tolist()

# Ajustement des longueurs
min_len = min(len(df_texts), len(labels))
df_texts, labels = df_texts[:min_len], labels[:min_len]

# Encodage avec Sentence-BERT
model = SentenceTransformer("all-MiniLM-L6-v2")
X_bert = model.encode(df_texts.tolist())

# Réduction de dimension
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_bert)

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["label"] = labels

# Visualisation
unique_labels = df_pca["label"].unique()
cmap = cm.get_cmap("tab20", len(unique_labels))

plt.figure(figsize=(12, 8))
for i, lab in enumerate(unique_labels):
    subset = df_pca[df_pca["label"] == lab]
    plt.scatter(subset["PC1"], subset["PC2"], color=cmap(i), alpha=0.8, label=lab)

plt.title("ACP sur les embeddings BERT - Couleurs par label")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc="best", fontsize=8)
plt.grid(True)
plt.show()
