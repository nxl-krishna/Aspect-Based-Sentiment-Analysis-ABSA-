import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""
# Phase 5: Transformer Model (BERT)

In Phase 4, we saw that adding an Attention Mechanism to a BiLSTM significantly improved accuracy (from ~69% to ~74%). However, recurrent models (like LSTMs) still have limitations. In this phase, we move to the state-of-the-art: **Transformers**.

## 1. Why Transformer Architectures Outperform LSTMs
- **Parallelization**: LSTMs process text sequentially word-by-word, which is slow and hard to parallelize. Transformers process the entire sequence at once.
- **No Vanishing Gradients over Long Contexts**: Even with gating and attention, LSTMs can struggle with very long dependencies. Transformers establish direct connections between any two words in a sequence regardless of their distance.

## 2. What is Self-Attention?
In Phase 4, we used *cross-attention* between the hidden states of the sentence and the specific aspect term vector. 
**Self-Attention** means every word in the sentence attends to every *other* word in the same sentence to compute its own representation. This allows the model to inherently understand syntax, coreference (e.g., resolving "it" to "the battery"), and context before it even looks at the specific aspect term.

## 3. The Power of Pre-training
Unlike our BiLSTM, which we initialized randomly or with static GloVe vectors and trained from scratch on 4,000 sentences, **BERT** (Bidirectional Encoder Representations from Transformers) is pre-trained on billions of words (Wikipedia + BookCorpus) using a Masked Language Modeling objective. 
This means BERT already understands English grammar and sentiment nuances deeply. We only need to "fine-tune" it on our small dataset.

## 4. Input Formatting for ABSA
BERT takes two sentences as input separated by a `[SEP]` token. For ABSA, we feed it:
`[CLS] The food was great but the service was terrible [SEP] service [SEP]`
BERT's self-attention layers will naturally cross-reference the word "service" with "terrible" during encoding!
"""))

cells.append(nbf.v4.new_code_cell("""
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# We use DistilBERT, a smaller, faster version of BERT that retains 97% of its performance.
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
"""))

cells.append(nbf.v4.new_code_cell("""
# 1. Load Data
df = pd.concat([pd.read_csv('../data/processed/Restaurants_Train.csv'), 
                pd.read_csv('../data/processed/Laptops_Train.csv')]).dropna(subset=['sentiment', 'text', 'aspect'])

# Map sentiment string to integer
sentiment_map = {'positive': 0, 'negative': 1, 'neutral': 2}
df['label'] = df['sentiment'].map(sentiment_map)

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# 2. Tokenize with Text Pair formatting: [CLS] Text [SEP] Aspect [SEP]
def tokenize_data(dataframe):
    return tokenizer(
        text=dataframe['text'].tolist(),
        text_pair=dataframe['aspect'].tolist(),
        padding='max_length',
        truncation=True,
        max_length=64, # short sequences are fine for this dataset
        return_tensors='pt'
    )

train_encodings = tokenize_data(train_df)
test_encodings = tokenize_data(test_df)
"""))

cells.append(nbf.v4.new_code_cell("""
# 3. Create PyTorch Dataset
class ABSA_HF_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ABSA_HF_Dataset(train_encodings, train_df['label'].tolist())
test_dataset = ABSA_HF_Dataset(test_encodings, test_df['label'].tolist())
"""))

cells.append(nbf.v4.new_code_cell("""
# 4. Initialize Pre-trained Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.to(device)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    _, _, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    return {
        'accuracy': acc,
        'macro_f1': f1
    }

# 5. Define Training Arguments
training_args = TrainingArguments(
    output_dir='../models/transformer_results',
    num_train_epochs=3,              # 3 epochs is standard for BERT fine-tuning
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none" # Disable wandb/etc
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)
"""))

cells.append(nbf.v4.new_code_cell("""
print("Starting fine-tuning...")
start_time = time.time()

# Train
trainer.train()

training_time = time.time() - start_time
print(f"Total Fine-Tuning Time: {training_time:.2f}s")
"""))

cells.append(nbf.v4.new_code_cell("""
# 6. Evaluate Best Model
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids

print(classification_report(labels, preds, target_names=['positive', 'negative', 'neutral']))

acc = accuracy_score(labels, preds)
_, _, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

# Append to baseline results
results_file = '../data/processed/baseline_results.csv'
if os.path.exists(results_file):
    results_df = pd.read_csv(results_file)
else:
    results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Macro F1', 'Training Time (s)'])

new_row = {'Model': 'DistilBERT (Fine-Tuned)', 'Accuracy': acc, 'Macro F1': f1, 'Training Time (s)': training_time}
results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
results_df.to_csv(results_file, index=False)

trainer.save_model('../models/best_absa_transformer')

results_df
"""))

nb['cells'] = cells

with open('notebooks/05_transformer.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Transformer notebook generated successfully.")
