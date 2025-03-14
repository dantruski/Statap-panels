import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import ssl
import nltk

# Désactiver la vérification SSL temporairement
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Télécharger les ressources nécessaires
nltk.download('stopwords')
nltk.download('wordnet')


corpus = [
    "Le roi gouverne le royaume avec force.",
    "Le roi est un homme puissant et courageux.",
    "La reine dirige le palais avec sagesse.",
    "La reine est une femme élégante et intelligente.",
    "L'homme travaille dur et défend ses idées.",
    "La femme apporte la douceur et la persévérance.",
    "Dans le passé, les rois et les reines étaient respectés.",
    "Le roi aime la guerre, mais la paix règne grâce à la reine."
]


nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words('french')  
lemmatizer = WordNetLemmatizer()  # Pour réduire les mots à leur forme "canonique".
 
 # Preprocessing du texte
def new_text(text):
    new_words = []
    words = text.split()
    for word in words :
        if word.lower() not in stop_words:
            new_words.append(lemmatizer.lemmatize(word.lower())) # réduction du mot à sa "racine".
    return ' '.join(new_words)

corpus = [new_text(doc) for doc in corpus]

# Génération d'embeddings via k-NN
class EmbeddingskNN:
    def __init__(self, n_neighbors=5):   
        self.vectorizer = TfidfVectorizer()  # Initialisation du vectoriseur TF-IDF
        self.matrix = None  # Matrice des coefficients TF-IDF
        self.words = None  # Liste des mots
        self.knn = NearestNeighbors(n_neighbors=n_neighbors)  # Initialisation du modèle k-NN
        
    # Entraînement du modèle
    def fit(self, corpus): 
        self.matrix = self.vectorizer.fit_transform(corpus)     # Matrice TF-IDF
        self.words = self.vectorizer.get_feature_names_out()  # Liste des mots du texte
        self.knn.fit(self.matrix.toarray().T)  # Entraînement du modèle k-NN. On travaille dans l'espace des mots donc on prend la transposée !!
        
    # Création de l'algorithme qui renvoie l'embedding d'un mot donné. On se base sur les kNN.
    def embbegin_from_word(self, word):
        if word in self.words:
            word_index = list(self.words).index(word)  
            word_vector = self.matrix[:, word_index].toarray().flatten().reshape(1, -1)  
            distances, indices = self.knn.kneighbors(word_vector)
            word_vectors = self.matrix.toarray().T  # Chaque ligne est un vecteur (dimension n_documents)
            neighbor_vectors = word_vectors[indices.flatten(), :]    
            center = np.mean(neighbor_vectors, axis=0)  
            dist_cent = word_vector.flatten() - center  
            embedding = dist_cent 
            return embedding
        else:
            raise ValueError(f"'{word}' n'apparait pas dans notre texte !")
        

# Exemples :

embedding_model = EmbeddingskNN(n_neighbors=6)
embedding_model.fit(corpus)
embedding_roi = embedding_model.embbegin_from_word('roi')
embedding_homme = embedding_model.embbegin_from_word('homme')
embedding_femme = embedding_model.embbegin_from_word('femme')
embedding_reine = embedding_model.embbegin_from_word('reine')

print("Embedding pour 'reine algébrique':", embedding_roi - embedding_homme + embedding_femme)
print("Embedding pour 'reine':", embedding_reine)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Création d'un dictionnaire pour stocker les embeddings
embeddings_dict = {
    "roi - homme + femme": embedding_roi - embedding_homme + embedding_femme,
    "reine": embedding_reine,
    "roi": embedding_roi,
    "homme": embedding_homme,
    "femme": embedding_femme
}

# Convertir les embeddings en une matrice NumPy
words = list(embeddings_dict.keys())  # Liste des mots
embeddings_matrix = np.array([embeddings_dict[word] for word in words])

# 🔹 Application de PCA pour réduire la dimension à 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(embeddings_matrix)

# 🔹 Création d'un DataFrame pour la visualisation
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["word"] = words  # Associer les mots aux points

# 🔹 Visualisation des résultats PCA
plt.figure(figsize=(10, 6))
plt.scatter(df_pca["PC1"], df_pca["PC2"], color="blue", alpha=0.7)

# Ajouter des annotations (mots) aux points
for i, word in enumerate(df_pca["word"]):
    plt.annotate(word, (df_pca["PC1"][i], df_pca["PC2"][i]), fontsize=12, ha='right')

plt.title("ACP sur les embeddings k-NN (TF-IDF)")
plt.xlabel("PC1 (Composante Principale 1)")
plt.ylabel("PC2 (Composante Principale 2)")
plt.grid(True)
plt.show()
