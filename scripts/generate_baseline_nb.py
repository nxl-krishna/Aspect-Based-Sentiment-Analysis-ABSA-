import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""
# Phase 2: Classical NLP Baseline

In this notebook, we establish a classical machine learning baseline for the Aspect-Based Sentiment Classification task.
We will use TF-IDF features and train Logistic Regression and Linear SVM models.

### Task Setup
Given a sentence and a specific aspect term, we want to predict the sentiment (positive, negative, neutral).
To incorporate the aspect information in a classical model, a common approach is to concatenate the aspect term with the sentence:
`Input = "[ASPECT] " + aspect_term + " [SEP] " + sentence`

This simple trick allows the TF-IDF vectorizer to capture the aspect explicitly alongside the sentence context.
"""))

cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import time

# Load data
df = pd.concat([pd.read_csv('../data/processed/Restaurants_Train.csv'), 
                pd.read_csv('../data/processed/Laptops_Train.csv')]).dropna(subset=['sentiment', 'text', 'aspect'])

# Create Train and Test split (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
"""))

cells.append(nbf.v4.new_code_cell("""
# Format inputs: prefix the aspect to the sentence
def format_input(row):
    return f"{row['aspect']} {row['text']}"

X_train = train_df.apply(format_input, axis=1)
y_train = train_df['sentiment']

X_test = test_df.apply(format_input, axis=1)
y_test = test_df['sentiment']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
"""))

cells.append(nbf.v4.new_code_cell("""
# 1. Logistic Regression
print("Training Logistic Regression...")
start_time = time.time()
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_model.fit(X_train_vec, y_train)
lr_time = time.time() - start_time

lr_preds = lr_model.predict(X_test_vec)
print(f"Logistic Regression Training Time: {lr_time:.3f}s")
print(classification_report(y_test, lr_preds))
"""))

cells.append(nbf.v4.new_code_cell("""
# 2. Linear SVM
print("Training Linear SVM...")
start_time = time.time()
svm_model = LinearSVC(max_iter=2000, class_weight='balanced', random_state=42)
svm_model.fit(X_train_vec, y_train)
svm_time = time.time() - start_time

svm_preds = svm_model.predict(X_test_vec)
print(f"Linear SVM Training Time: {svm_time:.3f}s")
print(classification_report(y_test, svm_preds))
"""))

cells.append(nbf.v4.new_code_cell("""
# Store results for model comparison table
def get_metrics(y_true, y_pred, model_name, t_time):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return {'Model': model_name, 'Accuracy': acc, 'Macro F1': f1, 'Training Time (s)': t_time}

results = []
results.append(get_metrics(y_test, lr_preds, 'TF-IDF + Logistic Regression', lr_time))
results.append(get_metrics(y_test, svm_preds, 'TF-IDF + SVM', svm_time))

results_df = pd.DataFrame(results)
results_df.to_csv('../data/processed/baseline_results.csv', index=False)
results_df
"""))

nb['cells'] = cells

with open('notebooks/02_baseline.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully.")
