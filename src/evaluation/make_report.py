"""Aggregate every run in reports/runs/ into reports/RESULTS.md, results.json and figures.

    python -m src.evaluation.make_report
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.semeval import DOMAINS
from src.training.utils import REPORTS_DIR, save_json

SENTIMENT_NAMES = {
    "deberta-v3-base": "DeBERTa-v3-base (fine-tuned)",
    "deberta-v3-base-weighted": "DeBERTa-v3-base + class-weighted loss",
    "distilbert": "DistilBERT (fine-tuned)",
}
SENTIMENT_ORDER = ["TF-IDF + Logistic Regression", "TF-IDF + Linear SVM", "BiLSTM", "BiLSTM + Attention",
                   "DistilBERT (fine-tuned)", "DeBERTa-v3-base + class-weighted loss", "DeBERTa-v3-base (fine-tuned)"]


def load_runs():
    sentiment, extraction = defaultdict(list), defaultdict(list)
    for path in sorted(glob.glob(os.path.join(REPORTS_DIR, "runs", "*.json"))):
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        if run["task"] == "sentiment-baselines":
            for r in run["results"]:
                sentiment[r["model"]].append({"test": r["test"], "train_time_s": r["train_time_s"], "seed": run["config"]["seed"]})
        elif run["task"] == "sentiment":
            sentiment[SENTIMENT_NAMES.get(run["config"]["name"], run["config"]["name"])].append(
                {"test": run["test"], "train_time_s": run["train_time_s"], "seed": run["config"]["seed"]})
        elif run["task"] == "extraction":
            extraction[f"{run['config']['name']} (BIO tagging)"].append(
                {"test": run["test"], "train_time_s": run["train_time_s"], "seed": run["config"]["seed"]})
    return sentiment, extraction


def mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def fmt(values, scale=100):
    m, s = mean_std(values)
    return f"{m * scale:.2f} ± {s * scale:.2f}" if len(values) > 1 and s > 1e-9 else f"{m * scale:.2f}"


def summarize(sentiment, extraction):
    summary = {"sentiment": {}, "extraction": {}}
    for model, runs in sentiment.items():
        summary["sentiment"][model] = {
            "seeds": [r["seed"] for r in runs],
            "train_time_s": mean_std([r["train_time_s"] for r in runs])[0],
            **{d: {k: mean_std([r["test"][d][k] for r in runs]) for k in ("accuracy", "macro_f1")} for d in DOMAINS},
            "neutral_f1": {d: mean_std([r["test"][d]["per_class"]["neutral"]["f1"] for r in runs]) for d in DOMAINS},
        }
    for model, runs in extraction.items():
        summary["extraction"][model] = {
            "seeds": [r["seed"] for r in runs],
            **{d: {k: mean_std([r["test"][d][k] for r in runs]) for k in ("precision", "recall", "f1")} for d in DOMAINS},
        }
    return summary


def plot_sentiment(sentiment, path):
    models = [m for m in SENTIMENT_ORDER if m in sentiment]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, metric, title in zip(axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro F1")):
        for i, domain in enumerate(DOMAINS):
            stats = [mean_std([r["test"][domain][metric] for r in sentiment[m]]) for m in models]
            ax.bar(x + (i - 0.5) * 0.38, [s[0] * 100 for s in stats], 0.38, yerr=[s[1] * 100 for s in stats],
                   label=domain, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" (fine-tuned)", "").replace(" + class-weighted loss", "\n+ weighted loss")
                            .replace("TF-IDF + ", "TF-IDF\n") for m in models], rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{title} on official SemEval-14 test sets")
        ax.set_ylim(40, 95)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("%")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_markdown(sentiment, extraction, path):
    lines = [
        "## Benchmark results",
        "",
        "All numbers are on the **official SemEval-2014 Task 4 test sets** (Restaurants: 1,120 aspects, "
        "Laptops: 638 aspects; conflict labels excluded, as is standard). Models are trained on the official "
        "train sets of both domains, with a stratified 10% validation split for epoch/model selection. "
        "Values are mean ± std over 3 random seeds (the TF-IDF models are deterministic). "
        "Training times are on a single RTX 3050 Laptop GPU (6 GB).",
        "",
        "### Aspect sentiment classification",
        "",
        "| Model | Restaurants Acc | Restaurants Macro-F1 | Laptops Acc | Laptops Macro-F1 | Neutral F1 (Rest / Lap) | Train time |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in [m for m in SENTIMENT_ORDER if m in sentiment] + [m for m in sentiment if m not in SENTIMENT_ORDER]:
        runs = sentiment[model]
        cells = [fmt([r["test"][d][k] for r in runs]) for d in DOMAINS for k in ("accuracy", "macro_f1")]
        neutral = " / ".join(f"{mean_std([r['test'][d]['per_class']['neutral']['f1'] for r in runs])[0] * 100:.1f}" for d in DOMAINS)
        t = mean_std([r["train_time_s"] for r in runs])[0]
        name = f"**{model}**" if model == SENTIMENT_ORDER[-1] else model
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {neutral} | {'<1s' if t < 1 else f'{t:.0f}s'} |")

    lines += [
        "",
        "### Aspect term extraction",
        "",
        "Exact-match span F1, the official SemEval-14 subtask 1 metric.",
        "",
        "| Model | Restaurants P | Restaurants R | Restaurants F1 | Laptops P | Laptops R | Laptops F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for model, runs in extraction.items():
        cells = [fmt([r["test"][d][k] for r in runs]) for d in DOMAINS for k in ("precision", "recall", "f1")]
        lines.append(f"| **DeBERTa-v3-base {model.split(' ', 1)[1]}** | " + " | ".join(cells) + " |")

    lines += ["", "![Sentiment benchmark](figures/sentiment_benchmark.png)", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    sentiment, extraction = load_runs()
    os.makedirs(os.path.join(REPORTS_DIR, "figures"), exist_ok=True)
    save_json(summarize(sentiment, extraction), os.path.join(REPORTS_DIR, "results.json"))
    plot_sentiment(sentiment, os.path.join(REPORTS_DIR, "figures", "sentiment_benchmark.png"))
    write_markdown(sentiment, extraction, os.path.join(REPORTS_DIR, "RESULTS.md"))
    print(open(os.path.join(REPORTS_DIR, "RESULTS.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
