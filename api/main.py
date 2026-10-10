from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import io
import sys
import os

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference.infer import ABSA_Inferencer, ABSAPipeline, AspectExtractor

# Global model instances (loaded on startup to save time)
inferencer = None
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global inferencer, pipeline
    try:
        inferencer = ABSA_Inferencer()
    except Exception as e:
        print(f"Error loading sentiment model: {e}")
    if inferencer is not None:
        try:
            pipeline = ABSAPipeline(sentiment=inferencer, extractor=AspectExtractor())
        except Exception as e:
            print(f"Aspect extraction unavailable: {e}")
    yield

app = FastAPI(
    title="Aspect-Based Sentiment Analysis API",
    description="Customer Feedback Intelligence API: aspect extraction + aspect-level sentiment.",
    version="2.0.0",
    lifespan=lifespan
)

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    aspect: str = Field(..., min_length=1)

class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1)

def _summary(results_df):
    return {
        "total_analyzed": len(results_df),
        "sentiment_breakdown": results_df['sentiment'].value_counts().to_dict(),
        "top_positive_aspects": results_df[results_df['sentiment'] == 'Positive']['aspect'].str.lower().value_counts().head(5).to_dict(),
        "top_negative_aspects": results_df[results_df['sentiment'] == 'Negative']['aspect'].str.lower().value_counts().head(5).to_dict(),
    }

@app.get("/health")
def health_check():
    if inferencer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {
        "status": "healthy",
        "sentiment_model": inferencer.model_name,
        "extractor_model": pipeline.extractor.model_name if pipeline else None,
    }

# Endpoints are sync (def) so FastAPI runs blocking model inference in a threadpool
# instead of stalling the event loop.
@app.post("/analyze")
def analyze(request: PredictRequest):
    """
    Analyze the sentiment of a specific aspect within a given text.
    """
    if inferencer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    
    try:
        return inferencer.predict(request.text, request.aspect)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-review")
def analyze_review(request: ReviewRequest):
    """
    Extract every aspect term mentioned in a review and classify the sentiment toward each one.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Aspect extraction model not loaded.")
    try:
        return pipeline.analyze(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-analyze")
def batch_analyze(file: UploadFile = File(...)):
    """
    Upload a CSV with a 'text' column. If it also has an 'aspect' column, those aspects are
    scored; otherwise aspects are extracted automatically.
    """
    if inferencer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        contents = file.file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8-sig')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if 'text' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain a 'text' column.")
    if 'aspect' not in df.columns and pipeline is None:
        raise HTTPException(status_code=400, detail="CSV must contain an 'aspect' column (aspect extraction model not loaded).")

    try:
        if 'aspect' in df.columns:
            results_df = inferencer.batch_predict(df)
        else:
            texts = df['text'].dropna().astype(str)
            texts = texts[texts.str.strip() != ''].tolist()
            rows = [
                {"text": r["text"], "aspect": a["aspect"], "sentiment": a["sentiment"], "confidence": a["confidence"]}
                for r in pipeline.analyze_batch(texts) for a in r["aspects"]
            ]
            results_df = pd.DataFrame(rows, columns=["text", "aspect", "sentiment", "confidence"])

        return {"summary": _summary(results_df), "results": results_df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
