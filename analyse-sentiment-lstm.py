import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("temp_dataset_with_source.csv", delimiter=";")
labels = df["satisfaction"].tolist()

# Combine conversation turns
conversations = df.drop(columns=["query_modifier_question", "answer", "labels", "satisfaction"]).apply(
    lambda row: [str(v) for v in row if pd.notna(v)], axis=1
).tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    conversations, labels, test_size=0.2, stratify=labels, random_state=42
)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Custom Dataset
class ConversationDataset(Dataset):
    def __init__(self, conversations, labels, tokenizer, max_turn_len=64, max_turns=10):
        self.conversations = conversations
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_turn_len = max_turn_len
        self.max_turns = max_turns

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        turns = self.conversations[idx][:self.max_turns]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        encoded_turns = [
            self.tokenizer(turn, padding='max_length', truncation=True,
                           max_length=self.max_turn_len, return_tensors="pt")
            for turn in turns
        ]

        input_ids = torch.cat([e["input_ids"] for e in encoded_turns], dim=0)       # (seq_len, max_len)
        attention_mask = torch.cat([e["attention_mask"] for e in encoded_turns], dim=0)

        return input_ids, attention_mask, label, len(turns)

# Collate function
def collate_fn(batch):
    input_ids, masks, labels, lengths = zip(*batch)
    lengths = torch.tensor(lengths)

    input_ids_padded = nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    masks_padded = nn.utils.rnn.pad_sequence(masks, batch_first=True)
    labels = torch.tensor(labels)

    return input_ids_padded, masks_padded, labels, lengths

# DataLoaders
train_dataset = ConversationDataset(X_train, y_train, tokenizer)
test_dataset = ConversationDataset(X_test, y_test, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)

# Model
class ConversationRNN(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=2):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.rnn = nn.LSTM(input_size=768, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask, lengths):
        B, S, L = input_ids.shape  # batch, seq_len, max_len
        input_ids = input_ids.view(B * S, L)
        attention_mask = attention_mask.view(B * S, L)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_tokens = outputs.last_hidden_state[:, 0, :]  # (B * S, 768)
        cls_tokens = cls_tokens.view(B, S, 768)

        packed = nn.utils.rnn.pack_padded_sequence(cls_tokens, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.rnn(packed)
        return self.fc(hidden[-1])

# Instantiate
model = ConversationRNN().to(device)
criterion = nn.CrossEntropyLoss()

# Optimizer with different LR for BERT
optimizer = torch.optim.Adam([
    {"params": model.bert.parameters(), "lr": 2e-5},
    {"params": model.rnn.parameters()},
    {"params": model.fc.parameters()}
], lr=1e-4)

import matplotlib.pyplot as plt

# Extend training loop for 10 epochs and track loss
num_epochs = 10
losses = []

print("Training...")
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for input_ids, masks, labels, lengths in train_loader:
        input_ids, masks, labels, lengths = input_ids.to(device), masks.to(device), labels.to(device), lengths.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, masks, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Plot the loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), losses, marker='o', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs. Epochs')
plt.legend()
plt.grid()
plt.show()