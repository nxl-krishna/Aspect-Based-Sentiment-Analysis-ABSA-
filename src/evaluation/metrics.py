from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.data.semeval import LABELS


def classification_metrics(y_true, y_pred, labels=LABELS):
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
        "confusion_matrix": {"labels": list(labels), "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
    }


def span_prf(gold, pred):
    """Exact-match precision/recall/F1 over spans.

    `gold` and `pred` are parallel lists (one entry per sentence) of span collections.
    """
    tp = n_gold = n_pred = 0
    for g, p in zip(gold, pred):
        g, p = set(map(tuple, g)), set(map(tuple, p))
        tp += len(g & p)
        n_gold += len(g)
        n_pred += len(p)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "n_gold": n_gold, "n_pred": n_pred, "tp": tp}
