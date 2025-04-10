import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import pandas as pd


#Jeu de données labellisé 
df=pd.read_csv("temp_dataset_with_source.csv", delimiter=";")
#Labels de satisfication
labels = df["satisfaction"].tolist()

conversations = df.drop(columns=["query_modifier_question", "answer","labels","satisfaction"]).apply(
    lambda row: " ".join(row.astype(str)), axis=1
).tolist()

# Séparation en train et test
X_train, X_test, y_train, y_test = train_test_split(
    conversations, labels, test_size=0.2, stratify=labels, random_state=42
)

#Chargement du modèle BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
bert.eval()

# Embedding avec BERT
@torch.no_grad()
def get_cls_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = bert(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token

# Classe de données personnalisée en torch
class ConversationDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        embedding = get_cls_embedding(text).squeeze(0)  # shape: (768,)
        return embedding, label

# Chargement des données
train_dataset = ConversationDataset(X_train, y_train)
test_dataset = ConversationDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# Modèle RNN
class ConversationRNN(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=2):
        super().__init__()
        self.rnn = nn.LSTM(input_size=768, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # x shape: (batch, 768)
        x = x.unsqueeze(1)  # Add time dimension: (batch, seq_len=1, 768)
        _, (hidden, _) = self.rnn(x)
        return self.fc(hidden[-1])


model = ConversationRNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training
print("Training model...")
model.train()
for epoch in range(3):
    for embeddings, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# Evaluation
print("\nEvaluating on test set...")
model.eval()
correct = total = 0
with torch.no_grad():
    for embeddings, targets in test_loader:
        outputs = model(embeddings)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.2%}")
