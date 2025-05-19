import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import BertTokenizer, BertModel

#Load BERT components
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()

#Load and preprocess your dataset
df = pd.read_json("temp_dataset.json")

# Split user and assistant turns from 'history'
def separate_history(history):
    user_content = ""
    assistant_content = ""
    for entry in history:
        if entry["role"] == "user":
            user_content = entry["content"]
        elif entry["role"] == "assistant":
            assistant_content = entry["content"]
    return pd.Series([user_content, assistant_content], index=["user", "assistant"])

# Process dataframe
df_copy = df["history"].apply(separate_history)
df_merged = pd.concat([df, df_copy], axis=1).drop(columns=["history"])

# Concatenate columns (excluding 'query_modifier_question' and 'answer')
conversations = df_merged.drop(columns=["query_modifier_question", "answer"]).apply(
    lambda row: " ".join(row.astype(str)), axis=1
).tolist()

#labels
#labels = [1 if i % 2 == 0 else 0 for i in range(len(conversations))]
labels = ...

#BERT CLS embedding extractor
@torch.no_grad()
def get_cls_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # (1, 768)

#Custom dataset class
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

# DataLoader
dataset = ConversationDataset(conversations, labels)
loader = DataLoader(dataset, batch_size=2)

# Simple MLP Classifier
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
for inputs, targets in loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print("Loss:", loss.item())
