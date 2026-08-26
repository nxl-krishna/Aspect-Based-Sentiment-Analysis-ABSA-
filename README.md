---
title: Customer Feedback Intelligence
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---
# Customer Feedback Intelligence: Aspect-Based Sentiment Analysis

An end-to-end Machine Learning pipeline that identifies specific aspects within customer reviews and extracts the sentiment directed at those exact aspects. Built from scratch moving from classical NLP to State-of-the-Art Transformers.

## Features
- **Raw Data Parsing**: Custom XML parsers for the SemEval-2014 Task 4 dataset.
- **Classical NLP**: TF-IDF and SVM baseline models.
- **Deep Learning**: Custom PyTorch Bidirectional LSTM with a mathematical Attention mechanism built from scratch.
- **Transformers**: HuggingFace implementation utilizing DeBERTa-v3 optimized for `[CLS] Text [SEP] Aspect [SEP]` tokenization.
- **Interpretability**: Extraction and visualization of Attention weights via Seaborn heatmaps for error analysis.
- **Production API**: A high-performance FastAPI REST backend.
- **Dashboard UI**: A sleek, dark-mode Streamlit interface providing single-review analysis and bulk CSV batch analytics with Plotly charting.

## Setup Instructions

1. **Activate the Virtual Environment**:
   ```powershell
   .\venv\Scripts\Activate
   ```

2. **Start the FastAPI Server**:
   ```powershell
   uvicorn api.main:app --reload
   ```
   The API will be available at `http://localhost:8000`.

3. **Start the Streamlit UI**:
   Open a second terminal, activate the `venv`, and run:
   ```powershell
   streamlit run frontend/app.py
   ```
   The dashboard will open automatically in your browser.

## Project Structure
- `api/`: FastAPI server and endpoints.
- `data/`: Raw XML datasets and processed CSVs.
- `frontend/`: Streamlit dashboard UI.
- `models/`: Saved PyTorch model weights.
- `notebooks/`: Research notebooks covering EDA, Baselines, LSTMs, Attention, Transformers, and Error Analysis.
- `scripts/`: Python scripts to generate and execute the research notebooks.
- `src/`: Modular Python packages for preprocessing and the inference pipeline.
- `INTERVIEW_GUIDE.md`: A comprehensive guide explaining the technical decisions and architecture of this project.

## Results
- **Baseline (Linear SVM)**: ~71.7% Accuracy
- **BiLSTM + Attention**: ~74.4% Accuracy
- **Transformer (BERT/DeBERTa)**: **~82.1% Accuracy**
