# Création d'embeddings maison à partir d'un MLP
import torch
import nltk
import re
import torch.optim as optim
import numpy as np
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

text = "Le roi gouverne le royaume avec force."
    

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words('french')  
lemmatizer = WordNetLemmatizer()  # Pour réduire les mots à leur forme "canonique".
 
 # Preprocessing du texte (comme pour avec les kNN)
def new_text(text):
    new_words = []
    text = re.sub(r'[^\w\s]', '', text)   # Supprime tout type de ponctuation
    words = text.split()
    for word in words :
        if word.lower() not in stop_words:
            new_words.append(lemmatizer.lemmatize(word.lower())) # réduction du mot à sa "racine".
    return ' '.join(new_words)

text = new_text(text)
print(text)

tokens = text.split()
numtokens = []
for token in tokens:
    numtoken = np.zeros(len(tokens))
    numtoken[tokens.index(token)] = 1
    numtokens.append(numtoken)
tokens = np.array(numtokens)


# Génération des embeddings via NLP
x = torch.from_numpy(tokens).float()
y = torch.tensor([
    [0.5, 0.2, 0.1, 0.8, 0.7],  # Embedding pour "roi"
    [0.3, 0.4, 0.6, 0.2, 0.5],  # Embedding pour "gouverne"
    [0.1, 0.9, 0.4, 0.3, 0.2],  # Embedding pour "le"
    [0.6, 0.1, 0.3, 0.7, 0.5]   # Embedding pour "royaume"
], dtype=torch.float32)

print(x.size())
print(y.size())
dim_in = 4
hidden_states = 10
dim_out = 5  

class EmbeddingsMLP(torch.nn.Module):   
    def __init__(self, dim_in, hidden_states, dim_out):
        super(EmbeddingsMLP, self).__init__()
        self.linear1 = torch.nn.Linear(dim_in, hidden_states)
        self.linear2 = torch.nn.Linear(hidden_states, dim_out)
        
    def forward(self,x):
        h_x = torch.relu(self.linear1(x))
        y_pred = self.linear2(h_x)
        return y_pred
    
model = EmbeddingsMLP(dim_in, hidden_states, dim_out)

L2_loss = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
for t in range(100):
    y_pred = model(x)
    loss = L2_loss(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(y_pred)

            

# Génération d'embeddings via un MLP (qu'on définit par héritage de torch.nn.Module)

    

