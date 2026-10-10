"""Fine-tune a transformer for aspect term extraction (BIO token classification).

Evaluated with exact-match span F1 on the official SemEval-14 test sets, the
standard metric for SemEval-14 subtask 1.

    python -m src.training.train_extraction --config configs/extraction_deberta.yaml
"""
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments)

from src.data.bio import BIO_LABELS, IGNORE, spans_to_token_labels, token_label_spans, token_labels_to_spans
from src.data.semeval import DOMAINS, load_sentences
from src.evaluation.metrics import span_prf
from src.training.utils import MODELS_DIR, REPORTS_DIR, load_config, save_json, set_seed

DEFAULTS = {
    "name": "deberta-v3-base",
    "model_name": "microsoft/deberta-v3-base",
    "epochs": 6,
    "lr": 3e-5,
    "batch_size": 16,
    "max_length": 128,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "bf16": True,
    "val_size": 0.1,
    "seed": 42,
    "save_model": True,
}


class BIODataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length):
        self.items = []
        for s in sentences:
            enc = tokenizer(s["text"], truncation=True, max_length=max_length, return_offsets_mapping=True)
            spans = [(a["from"], a["to"]) for a in s["aspects"]]
            enc["labels"] = spans_to_token_labels(s["text"], enc.pop("offset_mapping"), spans)
            self.items.append(dict(enc))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    gold, predicted = [], []
    for p_row, l_row in zip(preds, pred.label_ids):
        mask = l_row != IGNORE
        gold.append(token_label_spans(l_row[mask].tolist()))
        predicted.append(token_label_spans(p_row[mask].tolist()))
    m = span_prf(gold, predicted)
    return {"span_f1": m["f1"], "span_precision": m["precision"], "span_recall": m["recall"]}


def predict_spans(model, tokenizer, sentences, max_length, batch_size=64):
    model.eval()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(sentences), batch_size):
        texts = [s["text"] for s in sentences[i:i + batch_size]]
        enc = tokenizer(texts, truncation=True, max_length=max_length, padding=True,
                        return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping").tolist()
        with torch.no_grad():
            labels = model(**enc.to(device)).logits.argmax(-1).tolist()
        for text, off, lab in zip(texts, offsets, labels):
            out.append(token_labels_to_spans(text, off, lab))
    return out


def main():
    cfg = load_config(DEFAULTS)
    set_seed(cfg["seed"])
    run_name = f"extraction-{cfg['name']}-seed{cfg['seed']}"
    print(f"Run: {run_name}\nConfig: {cfg}")

    train_sents = [s for d in DOMAINS for s in load_sentences(d, "Train")]
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(train_sents))
    n_val = int(len(train_sents) * cfg["val_size"])
    val_sents = [train_sents[i] for i in idx[:n_val]]
    train_sents = [train_sents[i] for i in idx[n_val:]]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    # Some checkpoints (e.g. deberta-v3) are stored in fp16; training needs fp32 master weights
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["model_name"], num_labels=len(BIO_LABELS), dtype=torch.float32,
        id2label=dict(enumerate(BIO_LABELS)), label2id={l: i for i, l in enumerate(BIO_LABELS)},
    )

    bf16 = cfg["bf16"] and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, "checkpoints", run_name),
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
        metric_for_best_model="span_f1",
        bf16=bf16,
        logging_steps=50,
        seed=cfg["seed"],
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=BIODataset(train_sents, tokenizer, cfg["max_length"]),
        eval_dataset=BIODataset(val_sents, tokenizer, cfg["max_length"]),
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    train_time = time.time() - start

    results = {"run": run_name, "task": "extraction", "config": cfg, "train_time_s": train_time,
               "validation": trainer.evaluate(), "test": {}}
    for domain in DOMAINS:
        sents = load_sentences(domain, "Test")
        pred = predict_spans(trainer.model, tokenizer, sents, cfg["max_length"])
        gold = [[(a["from"], a["to"]) for a in s["aspects"]] for s in sents]
        results["test"][domain] = span_prf(gold, pred)
        r = results["test"][domain]
        print(f"{domain}: P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}")

    save_json(results, os.path.join(REPORTS_DIR, "runs", f"{run_name}.json"))
    if cfg["save_model"]:
        final_dir = os.path.join(MODELS_DIR, f"absa-extractor-{cfg['name']}")
        trainer.save_model(final_dir)
        print(f"Saved model to {final_dir}")


if __name__ == "__main__":
    main()
