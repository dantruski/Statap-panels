import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

# ----------- Préparation du texte -----------

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

stop_words = stopwords.words('french')  
lemmatizer = WordNetLemmatizer()

def new_text(text):
    return ' '.join([
        lemmatizer.lemmatize(word.lower()) 
        for word in text.split() 
        if word.lower() not in stop_words
    ])

corpus = [new_text(doc) for doc in corpus]

# ----------- Embeddings BERT -----------

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')

embeddings = []
for doc in corpus:
    inputs = tokenizer(doc, return_tensors="pt", padding=True, truncation=True)
    outputs = model_bert(**inputs)
    embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten())

embeddings = np.array(embeddings)
target_embeddings = torch.tensor(embeddings, dtype=torch.float32)

# ----------- TF-IDF + MLP -----------

vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(corpus).toarray()

class EmbeddingsMLP(nn.Module):   
    def __init__(self, dim_in, hidden_states, dim_out):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, hidden_states)
        self.linear2 = nn.Linear(hidden_states, dim_out)
    def forward(self,x):
        return self.linear2(torch.relu(self.linear1(x)))

model = EmbeddingsMLP(X.shape[1], 64, target_embeddings.shape[1])
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    inputs = torch.tensor(X, dtype=torch.float32) 
    outputs = model(inputs)
    loss = loss_fn(outputs, target_embeddings)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/1000, Loss: {loss.item()}")

# ----------- Embeddings MLP obtenus -----------

model.eval()
with torch.no_grad():
    embeddings_MLP = model(torch.tensor(X, dtype=torch.float32))

# ----------- t-SNE des phrases (MLP) -----------

sentences = [
    "Le roi gouverne le royaume",
    "Le roi est un homme puissant",
    "La reine dirige le palais",
    "La reine est une femme élégante",
    "L'homme travaille dur",
    "La femme apporte la douceur",
    "Dans le passé les rois et reines",
    "Le roi aime la guerre, la paix grâce à la reine"
]

tsne = TSNE(n_components=2, perplexity=3, random_state=0)
embeddings_2D = tsne.fit_transform(embeddings_MLP.detach().numpy())

plt.figure(figsize=(12, 8))
for i, sentence in enumerate(sentences):
    x, y = embeddings_2D[i]
    plt.scatter(x, y, s=50, color='steelblue')
    plt.annotate(sentence, (x, y), textcoords="offset points", xytext=(5, 2),
                 fontsize=9, ha='left',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
                 arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
plt.title("t-SNE des embeddings MLP (appris pour approximer BERT)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ----------- Relation vectorielle MLP -----------

roi = model(torch.tensor(X[0], dtype=torch.float32))
homme = model(torch.tensor(X[1], dtype=torch.float32))
femme = model(torch.tensor(X[3], dtype=torch.float32))
reine = model(torch.tensor(X[2], dtype=torch.float32))
embedding_ra = roi - homme + femme

embeddings_mots = torch.stack([roi, homme, femme, reine, embedding_ra])
mots = ["roi", "homme", "femme", "reine", "reine_algébrique"]

coords = TSNE(n_components=2, perplexity=3, random_state=0).fit_transform(embeddings_mots.detach().numpy())

plt.figure(figsize=(8, 6))
colors = {"roi": "blue", "homme": "green", "femme": "orange", "reine": "purple", "reine_algébrique": "red"}
for i, word in enumerate(mots):
    x, y = coords[i]
    plt.scatter(x, y, color=colors[word], s=80)
    plt.annotate(word, (x, y), textcoords="offset points", xytext=(5, 2),
                 fontsize=10, ha='left',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

# --- MLP vectorielle avec distances ---

def draw_arrow_mlp(start_word, end_word, color):
    i1, i2 = mots.index(start_word), mots.index(end_word)
    x1, y1 = coords[i1]
    x2, y2 = coords[i2]
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    
    # Tracer la flèche
    plt.arrow(x1, y1, dx, dy, head_width=0.5, color=color, alpha=0.6, length_includes_head=True)
    
    # Affichage de la distance
    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
    plt.text(xm, ym, f"{dist:.2f}", fontsize=9, color=color)
    return dist

d1_mlp = draw_arrow_mlp("homme", "femme", "gray")
d2_mlp = draw_arrow_mlp("roi", "reine_algébrique", "red")
d3_mlp = draw_arrow_mlp("reine_algébrique", "reine", "black")

plt.title("Relation vectorielle MLP avec distances", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print("\n--- Distances MLP ---")
print(f"homme → femme         : {d1_mlp:.4f}")
print(f"roi → reine_algébrique : {d2_mlp:.4f}")
print(f"reine_algébrique → reine : {d3_mlp:.4f}")


# ----------- Relation vectorielle BERT -----------

words_to_check = ["roi", "homme", "femme", "reine"]
word_embeddings_tensor = {}
for word in words_to_check:
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
    outputs = model_bert(**inputs)
    word_embeddings_tensor[word] = outputs.last_hidden_state.mean(dim=1).detach()

embedding_roi_bert = word_embeddings_tensor["roi"]
embedding_homme_bert = word_embeddings_tensor["homme"]
embedding_femme_bert = word_embeddings_tensor["femme"]
embedding_reine_bert = word_embeddings_tensor["reine"]
embedding_ra_bert = embedding_roi_bert - embedding_homme_bert + embedding_femme_bert

embeddings_bert = torch.cat([embedding_roi_bert, embedding_homme_bert, embedding_femme_bert, embedding_reine_bert, embedding_ra_bert], dim=0)
embeddings_bert = embeddings_bert.view(5, -1)
mots_bert = ["roi", "homme", "femme", "reine", "reine_algébrique"]

coords_bert = TSNE(n_components=2, perplexity=3, random_state=0).fit_transform(embeddings_bert.numpy())

plt.figure(figsize=(8, 6))
for i, word in enumerate(mots_bert):
    x, y = coords_bert[i]
    plt.scatter(x, y, color=colors[word], s=80)
    plt.annotate(word, (x, y), textcoords="offset points", xytext=(5, 2),
                 fontsize=10, ha='left',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

# --- BERT vectorielle avec distances ---

def draw_arrow_bert(start_word, end_word, color):
    i1, i2 = mots_bert.index(start_word), mots_bert.index(end_word)
    x1, y1 = coords_bert[i1]
    x2, y2 = coords_bert[i2]
    dx, dy = x2 - x1, y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    
    # Tracer la flèche
    plt.arrow(x1, y1, dx, dy, head_width=0.5, color=color, alpha=0.6, length_includes_head=True)
    
    # Affichage de la distance
    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
    plt.text(xm, ym, f"{dist:.2f}", fontsize=9, color=color)
    return dist

d1_bert = draw_arrow_bert("homme", "femme", "gray")
d2_bert = draw_arrow_bert("roi", "reine_algébrique", "red")
d3_bert = draw_arrow_bert("reine_algébrique", "reine", "black")

plt.title("Relation vectorielle BERT avec distances", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print("\n--- Distances BERT ---")
print(f"homme → femme         : {d1_bert:.4f}")
print(f"roi → reine_algébrique : {d2_bert:.4f}")
print(f"reine_algébrique → reine : {d3_bert:.4f}")
