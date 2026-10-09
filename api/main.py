from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import io
import sys
import os

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference.infer import ABSA_Inferencer

# Global inferencer instance (loaded on startup to save time)
inferencer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global inferencer
    try:
        # Load the HuggingFace model once on startup
        inferencer = ABSA_Inferencer()
    except Exception as e:
        print(f"Error loading model: {e}")
    yield

app = FastAPI(
    title="Aspect-Based Sentiment Analysis API",
    description="Customer Feedback Intelligence API powered by Transformers.",
    version="1.0.0",
    lifespan=lifespan
)

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    aspect: str = Field(..., min_length=1)

@app.get("/health")
def health_check():
    if inferencer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "healthy", "model": "loaded"}

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

@app.post("/batch-analyze")
def batch_analyze(file: UploadFile = File(...)):
    """
    Upload a CSV file containing 'text' and 'aspect' columns for batch processing.
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

    if 'text' not in df.columns or 'aspect' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain 'text' and 'aspect' columns.")

    try:
        results_df = inferencer.batch_predict(df)
        
        # Calculate aggregate analytics
        sentiment_counts = results_df['sentiment'].value_counts().to_dict()
        top_positive = results_df[results_df['sentiment'] == 'Positive']['aspect'].value_counts().head(5).to_dict()
        top_negative = results_df[results_df['sentiment'] == 'Negative']['aspect'].value_counts().head(5).to_dict()
        
        return {
            "summary": {
                "total_analyzed": len(results_df),
                "sentiment_breakdown": sentiment_counts,
                "top_positive_aspects": top_positive,
                "top_negative_aspects": top_negative
            },
            "results": results_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
