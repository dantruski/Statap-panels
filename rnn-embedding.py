import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertModel
from tensorflow.keras.datasets import imdb

from sklearn.model_selection import train_test_split


# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load pre-trained BERT model
model = BertModel.from_pretrained("bert-base-uncased")

# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)


df=pd.read_json('temp_dataset.json')
df.head()

def tokenize_function(df):
    return tokenizer(df["query_modifier_question"], padding=True, truncation=True, max_length=512)

tokenized_dataset = df.apply(tokenize_function, axis=1)

def extract_embedding(df):
    model.eval()
    inputs=tokenizer(df["query_modifier_question"], padding=True, truncation=True, max_length=512, return_tensors="pt")
   
    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[:, 0, :]
    cls_embeddings = outputs.pooler_output

    return {"embeddings": embeddings, "cls_embeddings": cls_embeddings}

embedding = df.apply(extract_embedding, axis=1)
print(embedding[0])
print(embedding[0]["embeddings"].shape)

