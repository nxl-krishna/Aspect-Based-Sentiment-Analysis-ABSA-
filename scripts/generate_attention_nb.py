import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""
# Phase 4: Attention Mechanism

In Phase 3, we noticed that a vanilla BiLSTM performs slightly worse than classical Machine Learning (TF-IDF + SVM) on our dataset. Why? 
Because a standard BiLSTM compresses the entire sentence into a single dense vector without knowing *what* to focus on. If a sentence has multiple aspects (e.g. "The **food** was great but the **service** was terrible"), the BiLSTM might get confused about which sentiment to output.

## 1. Why Attention?

Attention allows the neural network to dynamically assign different weights (importance) to different words in a sentence based on a specific "query" (in our case, the **aspect term**). Instead of forcing the LSTM to remember everything, we let it "attend" to the words that matter most for the given aspect.

## 2. Mathematical Formulation (Aspect-Based Attention)

Let $H = [h_1, h_2, ..., h_N]$ be the hidden states of the BiLSTM for a sentence of length $N$.
Let $v_a$ be the embedding representation of the aspect term (e.g., the average embedding of the aspect words).

1. **Alignment Score**: We compute an alignment score between each word's hidden state $h_i$ and the aspect vector $v_a$:
   $$ u_i = \\tanh(W_h h_i + W_a v_a + b) $$
   $$ e_i = w^T u_i $$

2. **Attention Weights**: We normalize the scores using a softmax function to get a probability distribution (the attention weights $\\alpha$):
   $$ \\alpha_i = \\frac{\\exp(e_i)}{\\sum_{j=1}^N \\exp(e_j)} $$

3. **Context Vector**: We compute the final sentence representation $r$ as a weighted sum of the BiLSTM hidden states:
   $$ r = \\sum_{i=1}^N \\alpha_i h_i $$

Finally, we pass this aspect-specific context vector $r$ to a linear layer to predict the sentiment.
"""))

cells.append(nbf.v4.new_code_cell("""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import time

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
print(f"Final Vocab Size: {vocab_size}")
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

class ABSA_Attention_Dataset(Dataset):
    def __init__(self, df, w2i, max_len=50):
        self.texts = []
        self.aspects = []
        self.labels = []
        
        for _, row in df.iterrows():
            text_encoded = encode_sequence(row['text_tokens'], w2i, max_len)
            # Aspect usually shorter, but we can encode it same way or take just the first few
            aspect_encoded = encode_sequence(row['aspect_tokens'], w2i, max_len=10)
            
            self.texts.append(text_encoded)
            self.aspects.append(aspect_encoded)
            self.labels.append(sentiment_map[row['sentiment']])
            
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.aspects[idx]), torch.tensor(self.labels[idx])

train_dataset = ABSA_Attention_Dataset(train_df, word2idx, max_len=MAX_LEN)
test_dataset = ABSA_Attention_Dataset(test_df, word2idx, max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
"""))

cells.append(nbf.v4.new_code_cell("""
class Attention_ABSA(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx, embedding_matrix):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        self.lstm = nn.LSTM(embed_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           batch_first=True,
                           dropout=dropout if n_layers > 1 else 0)
        
        # Attention Layers
        lstm_hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # W_h * H + W_a * v_a
        self.W_h = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        self.W_a = nn.Linear(embed_dim, lstm_hidden_dim)
        self.w = nn.Linear(lstm_hidden_dim, 1, bias=False)
        
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, aspect):
        # text: [batch_size, seq_len]
        # aspect: [batch_size, aspect_len]
        
        text_embedded = self.dropout(self.embedding(text))
        # text_embedded: [batch_size, seq_len, embed_dim]
        
        # We average the aspect embeddings to get a single aspect vector v_a
        aspect_embedded = self.embedding(aspect) # [batch_size, aspect_len, embed_dim]
        # Ignore PAD tokens in aspect when averaging (simplified here with simple mean)
        aspect_mask = (aspect != 0).float().unsqueeze(-1)
        v_a = (aspect_embedded * aspect_mask).sum(dim=1) / (aspect_mask.sum(dim=1) + 1e-8)
        # v_a: [batch_size, embed_dim]
        
        # Pass text through BiLSTM
        H, _ = self.lstm(text_embedded)
        # H: [batch_size, seq_len, lstm_hidden_dim]
        
        # --- ATTENTION MECHANISM ---
        # 1. Transform H and v_a
        transformed_H = self.W_h(H) # [batch_size, seq_len, lstm_hidden_dim]
        transformed_v_a = self.W_a(v_a).unsqueeze(1) # [batch_size, 1, lstm_hidden_dim]
        
        # 2. Compute non-linear scores
        M = torch.tanh(transformed_H + transformed_v_a) # [batch_size, seq_len, lstm_hidden_dim]
        
        # 3. Compute attention weights alpha
        scores = self.w(M).squeeze(-1) # [batch_size, seq_len]
        
        # Mask out padding in text before softmax
        padding_mask = (text == 0)
        scores = scores.masked_fill(padding_mask, -1e9)
        
        alpha = F.softmax(scores, dim=-1) # [batch_size, seq_len]
        
        # 4. Context Vector r (weighted sum of hidden states)
        r = torch.bmm(alpha.unsqueeze(1), H).squeeze(1) # [batch_size, lstm_hidden_dim]
        
        r = self.dropout(r)
        return self.fc(r)
"""))

cells.append(nbf.v4.new_code_cell("""
# Initialize Model
HIDDEN_DIM = 128
OUTPUT_DIM = 3
N_LAYERS = 1 # Keep it simple for attention
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = word2idx['<PAD>']

model = Attention_ABSA(vocab_size, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX, embedding_matrix)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

print(model)
"""))

cells.append(nbf.v4.new_code_cell("""
# Training Loop
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for texts, aspects, labels in iterator:
        texts, aspects, labels = texts.to(device), aspects.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(texts, aspects)
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
        for texts, aspects, labels in iterator:
            texts, aspects, labels = texts.to(device), aspects.to(device), labels.to(device)
            predictions = model(texts, aspects)
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
EPOCHS = 12
best_valid_loss = float('inf')

print("Starting training...")
start_time = time.time()

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc, preds, labels = evaluate(model, test_loader, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), '../models/attention_model.pt')
        
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')

training_time = time.time() - start_time
print(f"Total Training Time: {training_time:.2f}s")
"""))

cells.append(nbf.v4.new_code_cell("""
import os

# Load best model and evaluate
model.load_state_dict(torch.load('../models/attention_model.pt', weights_only=True))
_, _, preds, labels = evaluate(model, test_loader, criterion)

print(classification_report(labels, preds, target_names=['positive', 'negative', 'neutral']))

# Record metrics
acc = accuracy_score(labels, preds)
prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

# Append to baseline results
results_file = '../data/processed/baseline_results.csv'
if os.path.exists(results_file):
    results_df = pd.read_csv(results_file)
else:
    results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Macro F1', 'Training Time (s)'])

new_row = {'Model': 'BiLSTM + Attention', 'Accuracy': acc, 'Macro F1': f1, 'Training Time (s)': training_time}
results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
results_df.to_csv(results_file, index=False)

results_df
"""))

nb['cells'] = cells

with open('notebooks/04_attention.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Attention notebook generated successfully.")
