"""Publish the trained models to the Hugging Face Hub with model cards.

    hf auth login
    python scripts/push_to_hub.py --user <your-hf-username>

Then set these variables on the Space (Settings -> Variables) so the app loads them:
    ABSA_SENTIMENT_MODEL=<user>/absa-sentiment-deberta-v3
    ABSA_EXTRACTOR_MODEL=<user>/absa-extractor-deberta-v3
"""
import argparse
import json
import os

from huggingface_hub import HfApi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODELS = {
    "sentiment": {
        "local": "models/absa-sentiment-deberta-v3-base",
        "repo": "absa-sentiment-deberta-v3",
        "pipeline_tag": "text-classification",
        "title": "Aspect-level sentiment classifier (DeBERTa-v3-base, SemEval-2014)",
        "usage": (
            "tokenizer(review, aspect, return_tensors='pt')  # [CLS] review [SEP] aspect [SEP]\n"
            "labels: Negative / Neutral / Positive"
        ),
    },
    "extraction": {
        "local": "models/absa-extractor-deberta-v3-base",
        "repo": "absa-extractor-deberta-v3",
        "pipeline_tag": "token-classification",
        "title": "Aspect term extractor (DeBERTa-v3-base BIO tagger, SemEval-2014)",
        "usage": "pipeline('token-classification', model=..., aggregation_strategy='simple')  # labels: O / B-ASP / I-ASP",
    },
}


def metrics_table(kind):
    with open(os.path.join(ROOT, "reports", "results.json"), encoding="utf-8") as f:
        results = json.load(f)
    if kind == "sentiment":
        m = results["sentiment"]["DeBERTa-v3-base (fine-tuned)"]
        rows = [f"| {d} | {m[d]['accuracy'][0]:.2%} ± {m[d]['accuracy'][1]:.2%} | "
                f"{m[d]['macro_f1'][0]:.4f} ± {m[d]['macro_f1'][1]:.4f} |" for d in ("Restaurants", "Laptops")]
        return "| Test set | Accuracy | Macro-F1 |\n|---|---|---|\n" + "\n".join(rows)
    m = next(iter(results["extraction"].values()))
    rows = [f"| {d} | {m[d]['precision'][0]:.4f} | {m[d]['recall'][0]:.4f} | "
            f"{m[d]['f1'][0]:.4f} ± {m[d]['f1'][1]:.4f} |" for d in ("Restaurants", "Laptops")]
    return "| Test set | Precision | Recall | Span F1 (exact match) |\n|---|---|---|---|\n" + "\n".join(rows)


def model_card(kind, spec):
    return f"""---
language: en
license: mit
library_name: transformers
pipeline_tag: {spec['pipeline_tag']}
base_model: microsoft/deberta-v3-base
datasets:
- semeval2014-task4
tags:
- aspect-based-sentiment-analysis
- absa
---

# {spec['title']}

Fine-tuned from `microsoft/deberta-v3-base` on the SemEval-2014 Task 4 restaurant and laptop
training sets (both domains jointly). Part of an end-to-end ABSA system: an aspect extractor finds
the aspect terms in a review, and the sentiment classifier scores each one.

## Results on the official SemEval-2014 test sets

Mean ± std over 3 seeds; the uploaded checkpoint is the seed-42 run.

{metrics_table(kind)}

## Usage

```python
{spec['usage']}
```
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    for kind, spec in MODELS.items():
        local = os.path.join(ROOT, spec["local"])
        repo_id = f"{args.user}/{spec['repo']}"
        with open(os.path.join(local, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card(kind, spec))
        api.create_repo(repo_id, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=local, repo_id=repo_id, ignore_patterns=["training_args.bin"])
        print(f"Uploaded {local} -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
