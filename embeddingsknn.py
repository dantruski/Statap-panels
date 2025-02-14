import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


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
