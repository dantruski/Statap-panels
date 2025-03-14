import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # Import de TSNE

# 1. Charger le fichier labellisé (remplacez "labellised_data.json" par le nom de votre fichier)
with open("labellised_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Prétraitement des textes : on va concaténer les contenus de "history" pour chaque échange
def clean_text(text):
    text = text.lower()
    # Conserver lettres, espaces et caractères accentués
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", "", text)
    return text

texts = []
labels = []
for item in data:
    # Concaténation des "content" de chaque message dans l'historique
    conversation = " ".join([msg.get("content", "") for msg in item.get("history", [])])
    conversation = clean_text(conversation)
    texts.append(conversation)
    # Extraire le label
    labels.append(item.get("label", "unknown"))

# 3. Construction du vocabulaire
all_words = " ".join(texts).split()
vocab = set(all_words)
# On réserve l'indice 0 pour le padding
word2idx = {word: idx+1 for idx, word in enumerate(vocab)}
vocab_size = len(word2idx) + 1  # +1 pour le padding

# 4. Convertir les textes en séquences d'indices avec padding
max_len = max(len(text.split()) for text in texts)
def text_to_indices(text):
    tokens = text.split()
    indices = [word2idx[token] for token in tokens]
    # Padding jusqu'à la longueur max
    indices += [0] * (max_len - len(indices))
    return indices

X_indices = [text_to_indices(text) for text in texts]
X_tensor = torch.tensor(X_indices, dtype=torch.long)

# 5. Mapping des labels en indices numériques
unique_labels = sorted(list(set(labels)))
label2idx = {label: idx for idx, label in enumerate(unique_labels)}
y_indices = [label2idx[label] for label in labels]
y_tensor = torch.tensor(y_indices, dtype=torch.long)

# 6. Création d'un Dataset personnalisé
class ConversationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = ConversationDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 7. Définir le modèle avec une couche d'embedding et un MLP
embedding_dim = 50  # taille des vecteurs d'embedding

class EmbeddingClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, hidden1=20, hidden2=10):
        super(EmbeddingClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Agrégation par moyenne et MLP (deux couches cachées)
        self.fc1 = nn.Linear(embedding_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
    def forward(self, x):
        # x de taille (batch_size, max_len)
        embeds = self.embedding(x)  # (batch_size, max_len, embedding_dim)
        avg_embeds = embeds.mean(dim=1)  # Moyenne sur la dimension de la séquence
        x = F.relu(self.fc1(avg_embeds))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

num_classes = len(unique_labels)
model = EmbeddingClassifier(vocab_size, embedding_dim, num_classes)

# 8. Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 9. Entraîner le modèle
num_epochs = 20
losses = []
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)  # sorties de taille (batch_size, num_classes)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

# Affichage de la courbe d'apprentissage
plt.plot(range(1, num_epochs+1), losses)
plt.xlabel("Époch")
plt.ylabel("Perte")
plt.title("Courbe d'apprentissage du classificateur")
plt.show()

# 10. Tester le modèle sur quelques exemples
model.eval()
with torch.no_grad():
    sample_outputs = model(X_tensor[:5])
    predicted = torch.argmax(sample_outputs, dim=1)
    for i in range(5):
        print(f"Conversation (début): {texts[i][:100]}...")
        print(f"Label prédit : {unique_labels[predicted[i]]} - Label réel : {labels[i]}")
        print("-----")

# 11. Visualisation avec t-SNE
# Récupération des embeddings moyens pour toutes les conversations
with torch.no_grad():
    all_embeds = model.embedding(X_tensor)        # (nombre_samples, max_len, embedding_dim)
    avg_embeds = all_embeds.mean(dim=1)             # (nombre_samples, embedding_dim)
    avg_embeds_np = avg_embeds.cpu().numpy()        # conversion en numpy

# Application du t-SNE pour réduire à 2 dimensions
tsne = TSNE(n_components=2, random_state=42)
embeds_2d = tsne.fit_transform(avg_embeds_np)

# Visualisation du résultat
plt.figure(figsize=(8,6))
scatter = plt.scatter(embeds_2d[:,0], embeds_2d[:,1], c=y_tensor.numpy(), cmap='viridis', alpha=0.7)
plt.colorbar(scatter, ticks=range(num_classes), label='Labels')
plt.title("t-SNE des embeddings des conversations")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
