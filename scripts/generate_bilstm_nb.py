import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""
# Phase 3: Word Embeddings & BiLSTM Model

In this phase, we move from classical ML with sparse features (TF-IDF) to deep learning with dense distributed representations (Word Embeddings).

## 1. Word Embedding Experiment

**Why one-hot encoding is insufficient:**
One-hot encoding treats every word as completely independent. A vocabulary of size $V$ creates vectors of length $V$ where only one element is 1. It suffers from the curse of dimensionality and cannot capture semantic similarities (e.g., "terrible" and "awful" have orthogonal one-hot vectors).

**What embeddings represent:**
Word embeddings represent words as dense vectors in a continuous vector space where semantically similar words are mapped to nearby points. They capture semantic and syntactic relationships based on distributional semantics (words that appear in similar contexts share similar meanings).

**Static vs. Contextual Embeddings:**
Static embeddings (like Word2Vec, GloVe, FastText) assign a single fixed vector to a word regardless of its context (e.g., "bank" in "river bank" and "bank account" has the same vector). Contextual embeddings (like BERT) generate a representation dynamically based on the surrounding sentence.

**Word2Vec vs. FastText:**
- **Word2Vec** learns embeddings by predicting surrounding words (Skip-Gram) or predicting a target word from context (CBOW). It handles known words well but fails on Out-Of-Vocabulary (OOV) words.
- **FastText** extends Word2Vec by treating each word as composed of character n-grams. This allows it to generate embeddings for rare or OOV words by summing their n-gram embeddings.
"""))

cells.append(nbf.v4.new_code_cell("""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import time
from collections import Counter

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)
print(f"Using device: {device}")

try:
    nlp = spacy.load("en_core_web_md")
except:
    nlp = None
    print("Warning: spacy en_core_web_md not found.")

def tokenize(text):
    if nlp:
        return [token.text.lower() for token in nlp(str(text))]
    return str(text).lower().split()
"""))

cells.append(nbf.v4.new_code_cell("""
# 1. Load Data
df = pd.concat([pd.read_csv('../data/processed/Restaurants_Train.csv'), 
                pd.read_csv('../data/processed/Laptops_Train.csv')]).dropna(subset=['sentiment', 'text', 'aspect'])

# Create Train and Test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# Tokenize sentences and aspects
print("Tokenizing train texts...")
train_df['text_tokens'] = train_df['text'].apply(tokenize)
train_df['aspect_tokens'] = train_df['aspect'].apply(tokenize)

print("Tokenizing test texts...")
test_df['text_tokens'] = test_df['text'].apply(tokenize)
test_df['aspect_tokens'] = test_df['aspect'].apply(tokenize)
"""))

cells.append(nbf.v4.new_code_cell("""
# 2. Build Vocabulary and Embedding Matrix using SpaCy (GloVe)
print("Building vocabulary and embedding matrix...")
word2idx = {'<PAD>': 0, '<UNK>': 1}
idx2word = {0: '<PAD>', 1: '<UNK>'}

# en_core_web_md uses 300-d vectors
EMBED_DIM = 300
embedding_matrix = [np.zeros(EMBED_DIM), np.random.normal(scale=0.1, size=(EMBED_DIM,))]

# Build vocabulary from training data
vocab = set()
for tokens in train_df['text_tokens']:
    vocab.update(tokens)
for tokens in train_df['aspect_tokens']:
    vocab.update(tokens)

for word in sorted(vocab):
    idx = len(word2idx)
    word2idx[word] = idx
    idx2word[idx] = word
    
    if nlp and nlp.vocab.has_vector(word):
        embedding_matrix.append(nlp.vocab[word].vector)
    else:
        embedding_matrix.append(np.random.normal(scale=0.1, size=(EMBED_DIM,)))

embedding_matrix = np.array(embedding_matrix)
vocab_size = len(word2idx)
print(f"Final Vocab Size (with PAD/UNK): {vocab_size}")
"""))

cells.append(nbf.v4.new_code_cell("""
# 3. PyTorch Dataset preparation
sentiment_map = {'positive': 0, 'negative': 1, 'neutral': 2}
MAX_LEN = 50

def encode_sequence(tokens, w2i, max_len):
    seq = [w2i.get(t, w2i['<UNK>']) for t in tokens]
    if len(seq) < max_len:
        seq += [w2i['<PAD>']] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

class ABSADataset(Dataset):
    def __init__(self, df, w2i, max_len=50):
        self.texts = []
        self.labels = []
        
        for _, row in df.iterrows():
            # For BiLSTM baseline, we can append aspect tokens to text tokens to provide context
            # similar to how we did [ASPECT] + [TEXT] in Phase 2
            combined_tokens = row['aspect_tokens'] + ['<SEP>'] + row['text_tokens']
            encoded = encode_sequence(combined_tokens, w2i, max_len)
            self.texts.append(encoded)
            self.labels.append(sentiment_map[row['sentiment']])
            
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])

train_dataset = ABSADataset(train_df, word2idx, max_len=MAX_LEN)
test_dataset = ABSADataset(test_df, word2idx, max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
"""))

cells.append(nbf.v4.new_markdown_cell("""
## 2. BiLSTM Model

**Why LSTM instead of vanilla RNN?**
Vanilla RNNs suffer from the vanishing gradient problem when processing long sequences, meaning they forget early context. LSTMs introduce a memory cell and gating mechanisms (input, forget, output gates) that allow them to selectively remember and forget information over long sequences.

**Why bidirectional?**
In language, the meaning of a word often depends on both the words before it and the words after it. A Bidirectional LSTM processes the sequence in both forward and backward directions, concatenating the hidden states at each step. This provides a full left-to-right and right-to-left context.

**Why is BiLSTM useful for ABSA?**
In ABSA, the sentiment descriptor might appear *before* or *after* the aspect term (e.g., "The **battery** is *terrible*" vs "*Terrible* **battery**"). A BiLSTM captures these long-distance contextual dependencies perfectly from both sides.
"""))

cells.append(nbf.v4.new_code_cell("""
class BiLSTM_ABSA(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx, embedding_matrix):
        super().__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # Load pre-trained Word2Vec weights
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        # Optional: fine-tune embeddings or freeze them
        # self.embedding.weight.requires_grad = False 
        
        # BiLSTM Layer
        self.lstm = nn.LSTM(embed_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           batch_first=True,
                           dropout=dropout if n_layers > 1 else 0)
        
        # Classification Layer
        # Multiply hidden_dim by 2 because of bidirectional
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text: [batch_size, seq_len]
        
        embedded = self.dropout(self.embedding(text))
        # embedded: [batch_size, seq_len, embed_dim]
        
        output, (hidden, cell) = self.lstm(embedded)
        # hidden: [num_layers * num_directions, batch_size, hidden_dim]
        
        # Get the final hidden state of the forward and backward paths
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
            
        # hidden: [batch_size, hidden_dim * 2]
        return self.fc(hidden)
"""))

cells.append(nbf.v4.new_code_cell("""
# Initialize Model
HIDDEN_DIM = 128
OUTPUT_DIM = 3
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = word2idx['<PAD>']

model = BiLSTM_ABSA(vocab_size, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX, embedding_matrix)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print(model)
"""))

cells.append(nbf.v4.new_code_cell("""
# Training Loop
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for texts, labels in iterator:
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        
        preds = predictions.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in iterator:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts)
            loss = criterion(predictions, labels)
            
            preds = predictions.argmax(dim=1)
            acc = (preds == labels).float().mean()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator), all_preds, all_labels
"""))

cells.append(nbf.v4.new_code_cell("""
EPOCHS = 10
best_valid_loss = float('inf')

print("Starting training...")
start_time = time.time()

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc, preds, labels = evaluate(model, test_loader, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), '../models/bilstm_model.pt')
        
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')

training_time = time.time() - start_time
print(f"Total Training Time: {training_time:.2f}s")
"""))

cells.append(nbf.v4.new_code_cell("""
# Load best model and evaluate
model.load_state_dict(torch.load('../models/bilstm_model.pt', weights_only=True))
_, _, preds, labels = evaluate(model, test_loader, criterion)

print(classification_report(labels, preds, target_names=['positive', 'negative', 'neutral']))

# Record metrics
acc = accuracy_score(labels, preds)
prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

import os
# Append to baseline results
results_file = '../data/processed/baseline_results.csv'
if os.path.exists(results_file):
    results_df = pd.read_csv(results_file)
else:
    results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Macro F1', 'Training Time (s)'])

new_row = {'Model': 'BiLSTM', 'Accuracy': acc, 'Macro F1': f1, 'Training Time (s)': training_time}
results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
results_df.to_csv(results_file, index=False)

results_df
"""))

nb['cells'] = cells

with open('notebooks/03_bilstm.ipynb', 'w') as f:
    nbf.write(nb, f)

print("BiLSTM notebook generated successfully.")
