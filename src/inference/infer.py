import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import warnings

# Suppress HuggingFace warnings
warnings.filterwarnings("ignore")

class ABSA_Inferencer:
    """
    Production Inference Pipeline for Aspect-Based Sentiment Analysis.
    Loads a State-of-the-Art Transformer model optimized for ABSA.
    """
    def __init__(self, model_name="yangheng/deberta-v3-base-absa-v1.1"):
        print(f"Loading ABSA Transformer: {model_name}...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load State of the Art ABSA model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        # The DeBERTa ABSA model maps output logits to: 0=Negative, 1=Neutral, 2=Positive
        # We remap to our project standard
        self.id2label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        print("Model loaded successfully!")

    def _predict_pairs(self, texts, aspects):
        # The model natively accepts Text and Aspect separated by SEP token.
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

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
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

    def batch_predict(self, df: pd.DataFrame, batch_size: int = 32):
        """
        Predict sentiment for a batch of reviews in a DataFrame.
        Expects columns: ['text', 'aspect']
        """
        if 'text' not in df.columns or 'aspect' not in df.columns:
            raise ValueError("CSV must contain 'text' and 'aspect' columns.")

        df = df[['text', 'aspect']].dropna()
        df = df[(df['text'].astype(str).str.strip() != '') & (df['aspect'].astype(str).str.strip() != '')]
        texts = df['text'].astype(str).tolist()
        aspects = df['aspect'].astype(str).tolist()

        results = []
        for start in range(0, len(texts), batch_size):
            results.extend(self._predict_pairs(texts[start:start + batch_size], aspects[start:start + batch_size]))

        return pd.DataFrame(results, columns=["text", "aspect", "sentiment", "confidence"])

if __name__ == "__main__":
    # Simple test
    inferencer = ABSA_Inferencer()
    res = inferencer.predict("The battery life is incredible but the screen is too dim.", "battery life")
    print(res)
