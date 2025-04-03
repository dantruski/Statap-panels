import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


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
    def __init__(self, n_neighbors=9):   
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


# Choix du nombre de voisins optimal
## On va maintenant sélectionner l'hyperparamètre k qui donne les "meilleurs" embeddings. 
# Pour cela, nous allons utiliser le coefficient de silhouette qui, pour un mot donné, mesure la différence entre la distance moyenne entre le mot et ses voisins (même cluster) et la distance moyenne entre le mot et les mots des autres clusters.
# Le k optimal sera alors celui qui maximisera le coefficient de silhouette (relation sémantique plus forte).

def optimal_k_value(corpus):
    k_max = 10 # Nombre maximal de voisins
    scores = []  #Liste contenant les scores de silhouette (moyens) pour chaque k
    for k in range(2,k_max):
        model_knn = EmbeddingskNN(n_neighbors=k)
        model_knn.fit(corpus)
        embeddings = [model_knn.embbegin_from_word(word) for word in model_knn.words]

        kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)
        kmeans.fit(embeddings)   
        labels_found = kmeans.labels_
        
        scores.append(silhouette_score(embeddings, labels_found))
    
# On trace à présent le graphe associé (score en fonction de l'hyperparamètre k)

    plt.plot(range(2, k_max), scores, marker='o')
    plt.title('Choix du nombre optimal de voisins')
    plt.xlabel('Nombre de voisins')
    plt.ylabel('Silhouette Scores')
    plt.show()

optimal_k_value(corpus)

# 9 est la valeur optimale pour k. On entraîne donc désormais le modèle pour k=9.  

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Embeddings de mots
embedding_roi = embedding_model.embbegin_from_word('roi')
embedding_homme = embedding_model.embbegin_from_word('homme')
embedding_femme = embedding_model.embbegin_from_word('femme')
embedding_reine = embedding_model.embbegin_from_word('reine')
embedding_reine_alg = embedding_roi - embedding_homme + embedding_femme

# Stack pour t-SNE
word_vectors = np.vstack([
    embedding_roi,
    embedding_homme,
    embedding_femme,
    embedding_reine,
    embedding_reine_alg
])
word_labels = ['roi', 'homme', 'femme', 'reine', 'reine_algébrique']

# t-SNE
tsne = TSNE(n_components=2, perplexity=3, random_state=42)
coords = tsne.fit_transform(word_vectors)

# Affichage
plt.figure(figsize=(8, 6))
colors = {
    'roi': 'blue',
    'homme': 'green',
    'femme': 'orange',
    'reine': 'purple',
    'reine_algébrique': 'red'
}

for i, word in enumerate(word_labels):
    x, y = coords[i]
    plt.scatter(x, y, color=colors[word], s=80)
    plt.annotate(
        word,
        (x, y),
        textcoords="offset points",
        xytext=(5, 2),
        ha='left',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8)
    )

# Flèches avec mesure
def draw_arrow(start, end, color, label=None):
    i1 = word_labels.index(start)
    i2 = word_labels.index(end)
    x1, y1 = coords[i1]
    x2, y2 = coords[i2]
    dx, dy = x2 - x1, y2 - y1
    distance = np.sqrt(dx**2 + dy**2)
    
    # Tracer la flèche
    plt.arrow(x1, y1, dx, dy,
              head_width=0.02,
              length_includes_head=True,
              color=color,
              alpha=0.6)

    # Ajouter la distance au milieu de la flèche
    if label:
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        plt.text(xm, ym, f"{distance:.2f}", fontsize=9, color=color)

    return distance

# Affichage des flèches et mesures
d1 = draw_arrow("homme", "femme", "gray", "homme→femme")
d2 = draw_arrow("roi", "reine_algébrique", "red", "roi→reine_alg")
d3 = draw_arrow("reine_algébrique", "reine", "black", "reine_alg→reine")

plt.title("Relation vectorielle (k-NN TF-IDF) avec distances", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Résumé console
print("\n--- Distances entre vecteurs ---")
print(f"homme → femme         : {d1:.4f}")
print(f"roi → reine_algébrique : {d2:.4f}")
print(f"reine_algébrique → reine : {d3:.4f}")
