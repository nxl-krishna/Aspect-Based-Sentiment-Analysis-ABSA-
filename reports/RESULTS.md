## Benchmark results

All numbers are on the **official SemEval-2014 Task 4 test sets** (Restaurants: 1,120 aspects, Laptops: 638 aspects; conflict labels excluded, as is standard). Models are trained on the official train sets of both domains, with a stratified 10% validation split for epoch/model selection. Values are mean ± std over 3 random seeds (the TF-IDF models are deterministic). Training times are on a single RTX 3050 Laptop GPU (6 GB).

### Aspect sentiment classification

| Model | Restaurants Acc | Restaurants Macro-F1 | Laptops Acc | Laptops Macro-F1 | Neutral F1 (Rest / Lap) | Train time |
|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 72.41 | 60.15 | 57.68 | 53.12 | 38.7 / 42.0 | <1s |
| TF-IDF + Linear SVM | 71.70 | 57.72 | 59.87 | 53.29 | 32.8 / 39.0 | <1s |
| BiLSTM | 72.86 ± 0.95 | 60.30 ± 1.52 | 64.52 ± 1.19 | 57.16 ± 1.97 | 41.8 / 39.8 | 22s |
| BiLSTM + Attention | 74.14 ± 0.62 | 61.70 ± 1.86 | 65.36 ± 1.45 | 59.69 ± 1.29 | 40.4 / 48.2 | 13s |
| DistilBERT (fine-tuned) | 85.03 ± 0.61 | 77.08 ± 1.50 | 77.74 ± 0.56 | 73.56 ± 0.49 | 59.6 / 63.4 | 60s |
| DeBERTa-v3-base + class-weighted loss | 87.23 ± 0.58 | 80.04 ± 0.97 | 82.50 ± 0.27 | 78.86 ± 0.40 | 61.7 / 66.7 | 314s |
| **DeBERTa-v3-base (fine-tuned)** | 87.08 ± 0.29 | 79.80 ± 0.33 | 82.65 ± 0.45 | 79.06 ± 0.73 | 61.4 / 66.8 | 311s |

### Aspect term extraction

Exact-match span F1, the official SemEval-14 subtask 1 metric.

| Model | Restaurants P | Restaurants R | Restaurants F1 | Laptops P | Laptops R | Laptops F1 |
|---|---|---|---|---|---|---|
| **DeBERTa-v3-base (BIO tagging)** | 89.49 ± 0.56 | 92.06 ± 0.38 | 90.76 ± 0.45 | 86.02 ± 0.88 | 85.88 ± 0.50 | 85.95 ± 0.66 |

![Sentiment benchmark](figures/sentiment_benchmark.png)
