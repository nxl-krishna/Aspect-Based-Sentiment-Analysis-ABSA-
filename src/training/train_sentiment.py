"""Fine-tune a transformer for aspect-level sentiment classification.

Input is encoded as a sentence pair: [CLS] review [SEP] aspect [SEP].
Trains on the official SemEval-14 train sets (restaurants + laptops), selects the
best epoch on a stratified validation split, and reports on the official test sets.

    python -m src.training.train_sentiment --config configs/sentiment_deberta.yaml
"""
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.data.semeval import DOMAINS, LABEL2ID, LABELS, load_pairs
from src.evaluation.metrics import classification_metrics
from src.training.utils import MODELS_DIR, REPORTS_DIR, load_config, save_json, set_seed, split_train_val

DEFAULTS = {
    "name": "deberta-v3-base",
    "model_name": "microsoft/deberta-v3-base",
    "epochs": 5,
    "lr": 2e-5,
    "batch_size": 16,
    "max_length": 128,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "class_weights": False,
    "bf16": True,
    "val_size": 0.1,
    "seed": 42,
    "save_model": True,
}


class PairDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.enc = tokenizer(
            df["text"].tolist(), df["aspect"].tolist(),
            truncation="only_first", max_length=max_length,
        )
        self.labels = [LABEL2ID[s] for s in df["sentiment"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    m = classification_metrics([LABELS[i] for i in pred.label_ids], [LABELS[i] for i in preds])
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}


def main():
    cfg = load_config(DEFAULTS)
    set_seed(cfg["seed"])
    run_name = f"sentiment-{cfg['name']}-seed{cfg['seed']}"
    print(f"Run: {run_name}\nConfig: {cfg}")

    train_df, val_df = split_train_val(load_pairs("Train"), cfg["val_size"], 42, "sentiment")
    test_df = load_pairs("Test")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    # Some checkpoints (e.g. deberta-v3) are stored in fp16; training needs fp32 master weights
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(LABELS), dtype=torch.float32,
        id2label={i: l.capitalize() for i, l in enumerate(LABELS)},
        label2id={l.capitalize(): i for i, l in enumerate(LABELS)},
    )

    class_weights = None
    if cfg["class_weights"]:
        counts = train_df["sentiment"].map(LABEL2ID).value_counts().sort_index().values
        class_weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float)
        print(f"Class weights: {class_weights.tolist()}")

    bf16 = cfg["bf16"] and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    out_dir = os.path.join(MODELS_DIR, "checkpoints", run_name)
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=64,
        warmup_steps=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        bf16=bf16,
        logging_steps=50,
        seed=cfg["seed"],
        report_to="none",
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=PairDataset(train_df, tokenizer, cfg["max_length"]),
        eval_dataset=PairDataset(val_df, tokenizer, cfg["max_length"]),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    start = time.time()
    trainer.train()
    train_time = time.time() - start

    results = {"run": run_name, "task": "sentiment", "config": cfg, "train_time_s": train_time, "test": {}}
    results["validation"] = trainer.evaluate()
    for domain in DOMAINS:
        df = test_df[test_df["domain"] == domain]
        preds = trainer.predict(PairDataset(df, tokenizer, cfg["max_length"])).predictions.argmax(-1)
        results["test"][domain] = classification_metrics(df["sentiment"].tolist(), [LABELS[i] for i in preds])
        print(f"{domain}: acc={results['test'][domain]['accuracy']:.4f} macro_f1={results['test'][domain]['macro_f1']:.4f}")

    save_json(results, os.path.join(REPORTS_DIR, "runs", f"{run_name}.json"))
    if cfg["save_model"]:
        final_dir = os.path.join(MODELS_DIR, f"absa-sentiment-{cfg['name']}")
        trainer.save_model(final_dir)
        print(f"Saved model to {final_dir}")


if __name__ == "__main__":
    main()
