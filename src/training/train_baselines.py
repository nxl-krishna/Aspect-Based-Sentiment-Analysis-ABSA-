"""Train and evaluate the non-transformer sentiment baselines on the official test sets.

Models: TF-IDF + Logistic Regression, TF-IDF + Linear SVM, BiLSTM, BiLSTM + aspect attention.
Uses the same train/validation split as the transformer runs.

    python -m src.training.train_baselines --seed 42
"""
import os
import time

import numpy as np
import spacy
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.data.semeval import DOMAINS, LABEL2ID, LABELS, load_pairs
from src.evaluation.metrics import classification_metrics
from src.models.rnn import AspectAttentionClassifier, BiLSTMClassifier
from src.training.utils import MODELS_DIR, REPORTS_DIR, load_config, save_json, set_seed, split_train_val

DEFAULTS = {"seed": 42, "epochs": 12, "lr": 1e-3, "batch_size": 32, "max_len": 60, "aspect_len": 8, "val_size": 0.1}


def evaluate_on_test(test_df, predict_fn):
    return {
        d: classification_metrics(test_df[test_df["domain"] == d]["sentiment"].tolist(),
                                  predict_fn(test_df[test_df["domain"] == d]))
        for d in DOMAINS
    }


def run_classical(train_df, test_df, seed):
    def fmt(df):
        return (df["aspect"] + " " + df["text"]).tolist()

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train = vec.fit_transform(fmt(train_df))
    results = []
    for name, clf in [
        ("TF-IDF + Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ("TF-IDF + Linear SVM", LinearSVC(max_iter=5000, class_weight="balanced", random_state=seed)),
    ]:
        start = time.time()
        clf.fit(X_train, train_df["sentiment"])
        results.append({
            "model": name, "train_time_s": time.time() - start,
            "test": evaluate_on_test(test_df, lambda df: clf.predict(vec.transform(fmt(df))).tolist()),
        })
    return results


class Encoder:
    def __init__(self, train_df, nlp):
        self.nlp = nlp
        vocab = sorted({t for col in ("text", "aspect") for s in train_df[col] for t in self.tokenize(s)})
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SEP>": 2}
        for w in vocab:
            self.word2idx.setdefault(w, len(self.word2idx))
        rng = np.random.RandomState(0)
        dim = nlp.vocab.vectors_length
        self.embeddings = np.stack([
            np.zeros(dim) if i == 0 else
            (nlp.vocab[w].vector if nlp.vocab.has_vector(w) else rng.normal(scale=0.1, size=dim))
            for w, i in sorted(self.word2idx.items(), key=lambda kv: kv[1])
        ]).astype(np.float32)

    def tokenize(self, text):
        return [t.text.lower() for t in self.nlp.tokenizer(str(text))]

    def ids(self, tokens, max_len):
        seq = [self.word2idx.get(t, 1) for t in tokens][:max_len]
        return seq + [0] * (max_len - len(seq))

    def tensors(self, df, cfg, prepend_aspect):
        texts, aspects = [], []
        for text, aspect in zip(df["text"], df["aspect"]):
            a = self.tokenize(aspect)
            t = self.tokenize(text)
            texts.append(self.ids(a + ["<SEP>"] + t if prepend_aspect else t, cfg["max_len"]))
            aspects.append(self.ids(a, cfg["aspect_len"]))
        return torch.tensor(texts), torch.tensor(aspects)


def run_neural(name, model_cls, prepend_aspect, enc, train_df, val_df, test_df, cfg, device):
    set_seed(cfg["seed"])
    X_t, A_t = enc.tensors(train_df, cfg, prepend_aspect)
    y_t = torch.tensor([LABEL2ID[s] for s in train_df["sentiment"]])
    X_v, A_v = enc.tensors(val_df, cfg, prepend_aspect)
    y_v = [LABEL2ID[s] for s in val_df["sentiment"]]

    model = model_cls(enc.embeddings).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    def predict(X, A):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                logits, _ = model(X[i:i + 256].to(device), A[i:i + 256].to(device))
                preds.extend(logits.argmax(1).tolist())
        return preds

    best_f1, best_state = -1, None
    start = time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(X_t))
        for i in range(0, len(perm), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            opt.zero_grad()
            logits, _ = model(X_t[b].to(device), A_t[b].to(device))
            loss_fn(logits, y_t[b].to(device)).backward()
            opt.step()
        val = classification_metrics([LABELS[i] for i in y_v], [LABELS[i] for i in predict(X_v, A_v)])
        print(f"  {name} epoch {epoch + 1}: val acc={val['accuracy']:.4f} macro_f1={val['macro_f1']:.4f}")
        if val["macro_f1"] > best_f1:
            best_f1, best_state = val["macro_f1"], {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_time = time.time() - start

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_').replace('+', 'plus')}.pt"))

    def predict_df(df):
        X, A = enc.tensors(df, cfg, prepend_aspect)
        return [LABELS[i] for i in predict(X, A)]

    return {"model": name, "train_time_s": train_time, "validation_macro_f1": best_f1,
            "test": evaluate_on_test(test_df, predict_df)}


def main():
    cfg = load_config(DEFAULTS)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_df, val_df = split_train_val(load_pairs("Train"), cfg["val_size"], 42, "sentiment")
    test_df = load_pairs("Test")

    results = run_classical(train_df, test_df, cfg["seed"])
    nlp = spacy.load("en_core_web_md")
    enc = Encoder(train_df, nlp)
    results.append(run_neural("BiLSTM", BiLSTMClassifier, True, enc, train_df, val_df, test_df, cfg, device))
    results.append(run_neural("BiLSTM + Attention", AspectAttentionClassifier, False, enc, train_df, val_df, test_df, cfg, device))

    for r in results:
        print(r["model"], {d: f"acc={m['accuracy']:.4f} f1={m['macro_f1']:.4f}" for d, m in r["test"].items()})
    save_json({"task": "sentiment-baselines", "config": cfg, "results": results},
              os.path.join(REPORTS_DIR, "runs", f"sentiment-baselines-seed{cfg['seed']}.json"))


if __name__ == "__main__":
    main()
