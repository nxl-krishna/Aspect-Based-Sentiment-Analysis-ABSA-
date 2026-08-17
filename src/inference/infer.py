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

    def predict(self, text: str, aspect: str):
        """
        Predict sentiment for a single text-aspect pair.
        """
        # The model natively accepts Text and Aspect separated by SEP token
        inputs = self.tokenizer(
            text,
            text_pair=aspect,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_idx].item()
        
        return {
            "text": text,
            "aspect": aspect,
            "sentiment": self.id2label[pred_idx],
            "confidence": round(confidence * 100, 2)
        }

    def batch_predict(self, df: pd.DataFrame):
        """
        Predict sentiment for a batch of reviews in a DataFrame.
        Expects columns: ['text', 'aspect']
        """
        if 'text' not in df.columns or 'aspect' not in df.columns:
            raise ValueError("CSV must contain 'text' and 'aspect' columns.")
            
        results = []
        # For a production app, we would batch this. For demo purposes, sequential is fine.
        for _, row in df.iterrows():
            res = self.predict(str(row['text']), str(row['aspect']))
            results.append(res)
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Simple test
    inferencer = ABSA_Inferencer()
    res = inferencer.predict("The battery life is incredible but the screen is too dim.", "battery life")
    print(res)
