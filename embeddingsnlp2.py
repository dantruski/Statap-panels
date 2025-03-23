import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

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
            new_words.append(lemmatizer.lemmatize(word.lower())) 
    return ' '.join(new_words)

corpus = [new_text(doc) for doc in corpus]

# Création des embeddings cibles avec BERT 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')

embeddings = []
for doc in corpus:
    inputs = tokenizer(doc, return_tensors="pt", padding=True, truncation=True)
    outputs = model_bert(**inputs)  # Modifié : On utilise model_bert ici
    embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten())

embeddings = np.array(embeddings)
target_embeddings = torch.tensor(embeddings, dtype=torch.float32) 
print(target_embeddings)

# Création de la matrice TF-IDF 
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(corpus).toarray()
print(X.shape)

# Création des embeddings via MLP
class EmbeddingsMLP(torch.nn.Module):   
    def __init__(self, dim_in, hidden_states, dim_out):
        super(EmbeddingsMLP, self).__init__()
        self.linear1 = torch.nn.Linear(dim_in, hidden_states)
        self.linear2 = torch.nn.Linear(hidden_states, dim_out)
        
    def forward(self,x):
        h_x = torch.relu(self.linear1(x))
        y_pred = self.linear2(h_x)
        return y_pred
    
dim_in = X.shape[1]
hidden_states = 64
dim_out = target_embeddings.shape[1]  

# Modèle
model = EmbeddingsMLP(dim_in, hidden_states, dim_out)

# La fonction de perte (L^2)
loss_L2 = nn.MSELoss()  # Fonction de perte L^2
optimizer = optim.Adam(model.parameters(), lr=1e-6)

n_epochs = 1000

# Entraînement du modèle
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    inputs = torch.tensor(X, dtype=torch.float32) 
    outputs = model(inputs)  

    # Pour la perte, on utilise les embeddings cibles générés par BERT    
    target_embeddings = torch.tensor(embeddings, dtype=torch.float32)  
    loss = loss_L2(outputs, target_embeddings)  #LOSS MSE

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")

model.eval()   # passage en mode évaluation et arrêt de l'accumulation des gradients (phases inférence)
with torch.no_grad():   # pas de calcul de gradients
    embeddings_MLP = model(torch.tensor(X, dtype=torch.float32))

# Affichage des embeddings générés par le MLP


# VERIFICATION DE LA RELATION SEMANTIQUE : "roi" - "homme" + "femme" = "reine"
words_to_check = ["roi", "homme", "femme", "reine"]

# Préparation des embeddings spécifiques pour ces mots
word_embeddings = {}

for word in words_to_check:
    # Tokenization du mot
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
    outputs = model_bert(**inputs)  # Modifié : On utilise model_bert ici
    # Calcul de l'embedding
    word_embeddings[word] = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
    
word_embeddings_tensor = {word: torch.tensor(embedding, dtype=torch.float32) for word, embedding in word_embeddings.items()}

# Calcul des embeddings de "roi", "homme", "femme" et "reine" à partir du MLP
roi = model(torch.tensor(X[0], dtype=torch.float32))
homme = model(torch.tensor(X[1], dtype=torch.float32))
femme = model(torch.tensor(X[3], dtype=torch.float32))
reine = model(torch.tensor(X[2], dtype=torch.float32))

# Affichage des embeddings
print("Embedding pour roi", roi[:8])
print("Embedding pour reine", reine[:8])
print("Embedding pour homme", homme[:8])
print("Embedding pour femme", femme[:8])


