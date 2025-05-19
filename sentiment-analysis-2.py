#Importation des librairies
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split

#Chargement du modèle BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

#Jeu de données
df=pd.read_csv("temp_dataset_with_source.csv", delimiter=";")
#Labels de satisfication
labels = df["satisfaction"].tolist()

conversations = df.drop(columns=["query_modifier_question", "answer","labels","satisfaction"]).apply(
    lambda row: " ".join(row.astype(str)), axis=1
).tolist()

#Embedding avec BERT
@torch.no_grad()
def get_cls_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # (1, 768)

class ConversationDataset(Dataset):
    def __init__(self, conversations, labels):
        self.conversations = conversations
        self.labels = labels

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        embedding = get_cls_embedding(self.conversations[idx]).squeeze(0)  # (768,)
        label = torch.tensor(self.labels[idx])
        return embedding, label
    


# Assuming 'conversations' and 'labels' are defined
conversations_train, conversations_test, labels_train, labels_test = train_test_split(
    conversations, labels, test_size=0.2, random_state=42
)
# Create DataLoader for training and testing
train_dataset = ConversationDataset(conversations_train, labels_train)
test_dataset = ConversationDataset(conversations_test, labels_test)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


class ConversationClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # x: (batch_size, 768)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

# Build model
model = ConversationClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop (1 epoch demo)
model.train()
for inputs, targets in train_loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    #print("Loss:", loss.item())

#Display first 10 predicted labels
model.eval()
predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())

# Display first 10 predicted labels
print("Predicted labels for the first 10 samples in the test set:")
print(predictions[20:40])
# Display first 10 true labels
print("True labels for the first 10 samples in the test set:")
print(labels_test[20:40])



