# Customer Feedback Intelligence: Interview Guide

Talking points for explaining this project in an ML / data science interview. Every number here comes
from [reports/RESULTS.md](reports/RESULTS.md) and [reports/ERROR_ANALYSIS.md](reports/ERROR_ANALYSIS.md).

## 1. The 30-second pitch

"I built an end-to-end aspect-based sentiment analysis system. You paste a review like *'the pasta was
delicious but the waiter was rude'* and it returns *pasta: positive, waiter: negative*. It's two models I
fine-tuned myself: a DeBERTa-v3 token classifier that extracts aspect terms, and a DeBERTa-v3 sentence-pair
classifier that scores the sentiment toward each one. On the official SemEval-2014 test sets the extractor
gets 90.8 span F1 on restaurants and 86.0 on laptops, and the sentiment model gets 87.1% and 82.7% accuracy.
It's deployed as a Gradio app on Hugging Face Spaces and as a FastAPI service."

## 2. Why ABSA instead of normal sentiment analysis

A review is often mixed. Overall sentiment says "neutral"; ABSA says *what* is good and *what* is bad,
which is what a product or operations team can act on.

## 3. Data and evaluation protocol

- SemEval-2014 Task 4, restaurants + laptops: about 5,900 training aspects and 1,758 test aspects.
- I evaluate on the **official gold test sets** so the numbers are comparable to published work, and pick
  epochs on a stratified 10% validation split carved from train (never on test).
- Every model is run with **3 seeds** and reported as mean ± std. This mattered: the BiLSTM-vs-attention
  gap on restaurants turned out to be within seed noise.
- Conflict labels are excluded (standard practice).

## 4. Aspect extraction (the part most ABSA tutorials skip)

- Framed as BIO token classification (B-ASP, I-ASP, O).
- The labels come as character offsets, but the model sees subword tokens, so I project spans onto tokens
  with the tokenizer's offset mapping and decode predictions back to exact character spans.
- I verified the projection is lossless (100% round-trip on the test sets) before training, so any error
  is the model's, not the preprocessing's.
- Metric: exact-match span F1, the official subtask metric. Partial overlaps count as misses.

## 5. Aspect sentiment

- Input: `[CLS] review [SEP] aspect [SEP]`, so self-attention can link "rude" to "waiter".
- Truncation cuts only the review, never the aspect.
- The progression I benchmarked (Laptops accuracy): TF-IDF + LR 57.7 → BiLSTM 64.5 → BiLSTM + attention
  65.4 → DistilBERT 77.7 → DeBERTa-v3 82.7.
- The attention layer is implemented from scratch: `u_i = tanh(W_h h_i + W_a v_a)`, `alpha = softmax(w^T u)`,
  `r = sum alpha_i h_i`, where `v_a` is the mean aspect embedding and padding is masked before the softmax.

## 6. Error analysis (have this ready, it's the strongest signal of judgment)

- **Mixed-sentiment sentences are the hard case**: 74.0% accuracy vs 89.9% when all aspects share a sentiment.
- **Neutral is the bottleneck**: 68% of errors are neutral aspects predicted as positive or negative.
- Many confident "errors" are wishes or descriptions: "I *needed* a laptop with a nice screen" (gold:
  neutral, predicted: positive). The model reacts to "nice" without recognising the hypothetical.
- **I tested a fix and it didn't work**: class-weighted loss changed neutral F1 by under 0.5 points across
  3 seeds. Conclusion: the problem is ambiguity, not class imbalance. Next steps I'd try: data augmentation
  with hypothetical/irrealis sentences, or more context than one sentence.

## 7. A debugging story

"My first DeBERTa run predicted the majority class for everything. I ruled out the data pipeline by
running DistilBERT through the same script, which learned fine. Then I ran a 64-example overfitting test:
DeBERTa's loss went to NaN within 8 steps. The cause was that Transformers 5 keeps the checkpoint's stored
dtype, and deberta-v3-base is stored in fp16, so it was training in pure fp16 with no fp32 master weights.
Loading in fp32 and using bf16 autocast fixed it."

## 8. Engineering

- YAML configs, fixed seeds, JSON results per run, one script to reproduce everything.
- 20 pytest tests (BIO edge cases, metrics, parsing, every API endpoint with fake models) in GitHub Actions.
- FastAPI endpoints are sync so blocking inference runs in a threadpool instead of stalling the event loop;
  proper 4xx vs 5xx errors; batched GPU inference.
- Latency: 59 ms per review on an RTX 3050 laptop GPU, 121 reviews/s batched, 158 ms on CPU.

## 9. Likely questions

- **Why DeBERTa over BERT/RoBERTa?** Disentangled attention (separate content and position embeddings) and
  a stronger pre-training objective; it's consistently ahead on GLUE-style benchmarks at the same size.
  I'd confirm with an ablation if I had more time; DistilBERT is the comparison I did run (−5 points on laptops).
- **Why are laptops harder?** More technical vocabulary and more neutral, descriptive mentions
  ("the screen is 15 inches"); the non-transformer models drop much more on laptops than on restaurants.
- **What would you do next?** Joint extraction + sentiment in one model, implicit aspects, domain adaptation
  to new product categories, and quantization/ONNX for cheaper CPU serving.
