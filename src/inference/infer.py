import os
import warnings

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

from src.data.bio import token_labels_to_spans

# Suppress HuggingFace warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_SENTIMENT_MODEL = os.path.join(PROJECT_ROOT, "models", "absa-sentiment-deberta-v3-base")
LOCAL_EXTRACTOR_MODEL = os.path.join(PROJECT_ROOT, "models", "absa-extractor-deberta-v3-base")
HUB_SENTIMENT_MODEL = "nxl-krishna/absa-sentiment-deberta-v3"
HUB_EXTRACTOR_MODEL = "nxl-krishna/absa-extractor-deberta-v3"
FALLBACK_SENTIMENT_MODEL = "yangheng/deberta-v3-base-absa-v1.1"


def model_candidates(env_var, local_dir, *hub_ids):
    """Model locations in priority order: environment variable > local directory > Hub repos."""
    candidates = []
    if os.environ.get(env_var):
        candidates.append(os.environ[env_var])
    if os.path.isdir(local_dir):
        candidates.append(local_dir)
    candidates.extend(hub_ids)
    return candidates


def _load_first(candidates, model_cls):
    errors = []
    for name in candidates:
        try:
            return name, AutoTokenizer.from_pretrained(name), model_cls.from_pretrained(name)
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise OSError("Could not load any model:\n" + "\n".join(errors))


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ABSA_Inferencer:
    """
    Aspect-level sentiment classifier.
    Input is encoded as a sentence pair: [CLS] review [SEP] aspect [SEP].
    """
    def __init__(self, model_name=None):
        candidates = [model_name] if model_name else model_candidates(
            "ABSA_SENTIMENT_MODEL", LOCAL_SENTIMENT_MODEL, HUB_SENTIMENT_MODEL, FALLBACK_SENTIMENT_MODEL)
        self.model_name, self.tokenizer, self.model = _load_first(candidates, AutoModelForSequenceClassification)
        print(f"Loaded ABSA sentiment model: {self.model_name}")
        self.device = _device()
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {int(i): label.capitalize() for i, label in self.model.config.id2label.items()}

    def _predict_pairs(self, texts, aspects):
        # truncation="only_first" so a long review never cuts off the aspect.
        inputs = self.tokenizer(
            texts,
            text_pair=aspects,
            return_tensors="pt",
            padding=True,
            truncation="only_first",
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits.float(), dim=-1)
        confidences, pred_idxs = probs.max(dim=-1)

        return [
            {
                "text": text,
                "aspect": aspect,
                "sentiment": self.id2label[idx],
                "confidence": round(conf * 100, 2)
            }
            for text, aspect, idx, conf in zip(texts, aspects, pred_idxs.tolist(), confidences.tolist())
        ]

    def predict(self, text: str, aspect: str):
        """
        Predict sentiment for a single text-aspect pair.
        """
        return self._predict_pairs([text], [aspect])[0]

    def predict_pairs(self, texts, aspects, batch_size: int = 32):
        results = []
        for start in range(0, len(texts), batch_size):
            results.extend(self._predict_pairs(texts[start:start + batch_size], aspects[start:start + batch_size]))
        return results

    def batch_predict(self, df: pd.DataFrame, batch_size: int = 32):
        """
        Predict sentiment for a batch of reviews in a DataFrame.
        Expects columns: ['text', 'aspect']
        """
        if 'text' not in df.columns or 'aspect' not in df.columns:
            raise ValueError("CSV must contain 'text' and 'aspect' columns.")

        df = df[['text', 'aspect']].dropna()
        df = df[(df['text'].astype(str).str.strip() != '') & (df['aspect'].astype(str).str.strip() != '')]
        results = self.predict_pairs(df['text'].astype(str).tolist(), df['aspect'].astype(str).tolist(), batch_size)
        return pd.DataFrame(results, columns=["text", "aspect", "sentiment", "confidence"])


class AspectExtractor:
    """Finds aspect terms in a review with a BIO token-classification model."""

    def __init__(self, model_name=None):
        candidates = [model_name] if model_name else model_candidates(
            "ABSA_EXTRACTOR_MODEL", LOCAL_EXTRACTOR_MODEL, HUB_EXTRACTOR_MODEL)
        self.model_name, self.tokenizer, self.model = _load_first(candidates, AutoModelForTokenClassification)
        print(f"Loaded aspect extraction model: {self.model_name}")
        self.device = _device()
        self.model.to(self.device)
        self.model.eval()

    def extract(self, texts, batch_size: int = 32):
        """Return, for each text, a list of {"aspect", "from", "to"} dicts."""
        results = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = self.tokenizer(batch, truncation=True, max_length=128, padding=True,
                                 return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping").tolist()
            with torch.no_grad():
                labels = self.model(**enc.to(self.device)).logits.argmax(-1).tolist()
            for text, off, lab in zip(batch, offsets, labels):
                results.append([
                    {"aspect": text[s:e], "from": s, "to": e}
                    for s, e in token_labels_to_spans(text, off, lab)
                ])
        return results


class ABSAPipeline:
    """End-to-end ABSA: extract aspect terms, then classify the sentiment toward each one."""

    def __init__(self, sentiment=None, extractor=None):
        self.sentiment = sentiment or ABSA_Inferencer()
        self.extractor = extractor or AspectExtractor()

    def analyze_batch(self, texts, batch_size: int = 32):
        texts = [str(t) for t in texts]
        extracted = self.extractor.extract(texts, batch_size)
        pair_texts = [t for t, aspects in zip(texts, extracted) for _ in aspects]
        pair_aspects = [a["aspect"] for aspects in extracted for a in aspects]
        sentiments = iter(self.sentiment.predict_pairs(pair_texts, pair_aspects, batch_size))

        results = []
        for text, aspects in zip(texts, extracted):
            for a in aspects:
                s = next(sentiments)
                a.update(sentiment=s["sentiment"], confidence=s["confidence"])
            results.append({"text": text, "aspects": aspects})
        return results

    def analyze(self, text: str):
        return self.analyze_batch([text])[0]


if __name__ == "__main__":
    pipeline = ABSAPipeline()
    print(pipeline.analyze("The battery life is incredible but the screen is too dim."))
