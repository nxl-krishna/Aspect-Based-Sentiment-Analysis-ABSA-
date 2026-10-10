import argparse
import json
import os
import random

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split

from src.data.semeval import PROJECT_ROOT

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(defaults):
    """Merge defaults <- YAML file (--config) <- CLI overrides (--key value)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    for key, value in defaults.items():
        kind = type(value) if value is not None else str
        if kind is bool:
            parser.add_argument(f"--{key}", type=lambda s: s.lower() in ("1", "true", "yes"))
        else:
            parser.add_argument(f"--{key}", type=kind)
    args = parser.parse_args()

    config = dict(defaults)
    if args.config:
        with open(args.config, encoding="utf-8-sig") as f:
            config.update(yaml.safe_load(f) or {})
    config.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
    return config


def split_train_val(df, val_size, seed, stratify_col=None):
    return train_test_split(
        df, test_size=val_size, random_state=seed,
        stratify=df[stratify_col] if stratify_col else None,
    )


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
