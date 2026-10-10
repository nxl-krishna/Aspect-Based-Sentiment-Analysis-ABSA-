"""Error analysis of the deployed sentiment model on the official SemEval-14 test sets.

Writes reports/ERROR_ANALYSIS.md, reports/figures/confusion_matrices.png and
reports/error_analysis.json.

    python -m src.evaluation.error_analysis
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.semeval import DOMAINS, LABELS, load_pairs
from src.evaluation.metrics import classification_metrics
from src.inference.infer import ABSA_Inferencer
from src.training.utils import REPORTS_DIR, save_json


def sentence_type(sentiments):
    if len(sentiments) == 1:
        return "single aspect"
    return "multi-aspect, mixed sentiment" if sentiments.nunique() > 1 else "multi-aspect, same sentiment"


def main():
    test = load_pairs("Test")
    model = ABSA_Inferencer()
    preds = model.predict_pairs(test["text"].tolist(), test["aspect"].tolist(), batch_size=64)
    test["pred"] = [p["sentiment"].lower() for p in preds]
    test["confidence"] = [p["confidence"] for p in preds]
    test["correct"] = test["pred"] == test["sentiment"]
    test["sentence_type"] = test.groupby(["domain", "text"])["sentiment"].transform(sentence_type)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    report = {"model": model.model_name, "domains": {}}
    for ax, domain in zip(axes, DOMAINS):
        d = test[test["domain"] == domain]
        m = classification_metrics(d["sentiment"].tolist(), d["pred"].tolist())
        report["domains"][domain] = m
        cm = np.array(m["confusion_matrix"]["matrix"])
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{domain} (acc {m['accuracy']:.1%}, macro-F1 {m['macro_f1']:.3f})")
    fig.tight_layout()
    os.makedirs(os.path.join(REPORTS_DIR, "figures"), exist_ok=True)
    fig.savefig(os.path.join(REPORTS_DIR, "figures", "confusion_matrices.png"), dpi=130)
    plt.close(fig)

    by_type = (test.groupby("sentence_type")["correct"].agg(["mean", "size"])
               .rename(columns={"mean": "accuracy", "size": "n"}).sort_values("accuracy", ascending=False))
    report["accuracy_by_sentence_type"] = {k: {"accuracy": float(v["accuracy"]), "n": int(v["n"])} for k, v in by_type.iterrows()}

    errors = test[~test["correct"]]
    pair_counts = errors.groupby(["sentiment", "pred"]).size().sort_values(ascending=False)
    report["error_pairs"] = {f"{t} -> {p}": int(n) for (t, p), n in pair_counts.items()}
    confident = errors.sort_values("confidence", ascending=False).head(12)

    lines = [
        "## Error analysis: DeBERTa-v3 aspect sentiment model",
        "",
        f"Evaluated on the official SemEval-2014 test sets ({len(test)} aspects). "
        f"Overall accuracy {test['correct'].mean():.1%}; {len(errors)} errors.",
        "",
        "![Confusion matrices](figures/confusion_matrices.png)",
        "",
        "### Accuracy by sentence type",
        "",
        "| Sentence type | Aspects | Accuracy |",
        "|---|---|---|",
        *[f"| {k} | {int(v['n'])} | {v['accuracy']:.1%} |" for k, v in by_type.iterrows()],
        "",
        "### Most common confusions",
        "",
        "| True -> Predicted | Count | Share of errors |",
        "|---|---|---|",
        *[f"| {t} -> {p} | {n} | {n / len(errors):.0%} |" for (t, p), n in pair_counts.items()],
        "",
        "### Most confident mistakes",
        "",
        "| Domain | Sentence | Aspect | True | Predicted | Confidence |",
        "|---|---|---|---|---|---|",
        *[f"| {r.domain} | {r.text.replace('|', '/')} | {r.aspect} | {r.sentiment} | {r.pred} | {r.confidence:.1f}% |"
          for r in confident.itertuples()],
        "",
    ]
    with open(os.path.join(REPORTS_DIR, "ERROR_ANALYSIS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    save_json(report, os.path.join(REPORTS_DIR, "error_analysis.json"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
