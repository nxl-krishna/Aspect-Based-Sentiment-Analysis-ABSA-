import spaces
import html
import gradio as gr
import pandas as pd
import plotly.express as px
import sys
import os

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.inference.infer import ABSA_Inferencer, ABSAPipeline, AspectExtractor

SENTIMENT_COLORS = {'Positive': '#4caf50', 'Negative': '#f44336', 'Neutral': '#9e9e9e'}
RESULTS_MD = os.path.join(os.path.dirname(__file__), "reports", "RESULTS.md")

print("Loading models (this might take a minute on startup)...")
try:
    inferencer = ABSA_Inferencer()
except Exception as e:
    print(f"Error loading sentiment model: {e}")
    inferencer = None

pipeline = None
if inferencer is not None:
    try:
        pipeline = ABSAPipeline(sentiment=inferencer, extractor=AspectExtractor())
    except Exception as e:
        print(f"Aspect extraction unavailable: {e}")

EMPTY_ASPECTS = pd.DataFrame(columns=["Aspect", "Sentiment", "Confidence (%)"])


def _highlight(text, aspects):
    segments, cursor = [], 0
    for a in sorted(aspects, key=lambda a: a["from"]):
        if a["from"] > cursor:
            segments.append((text[cursor:a["from"]], None))
        segments.append((text[a["from"]:a["to"]], a["sentiment"]))
        cursor = a["to"]
    if cursor < len(text):
        segments.append((text[cursor:], None))
    return segments


@spaces.GPU
def analyze_review(text):
    if pipeline is None:
        return [("Aspect extraction model is not loaded.", None)], EMPTY_ASPECTS
    if not text or not text.strip():
        return [("Please enter a review.", None)], EMPTY_ASPECTS
    result = pipeline.analyze(text)
    if not result["aspects"]:
        return [(text, None)], EMPTY_ASPECTS
    table = pd.DataFrame([
        {"Aspect": a["aspect"], "Sentiment": a["sentiment"], "Confidence (%)": a["confidence"]}
        for a in result["aspects"]
    ])
    return _highlight(text, result["aspects"]), table


@spaces.GPU
def analyze_single(text, aspect):
    if not inferencer:
        return "Error: Model failed to load."
    if not text or not text.strip() or not aspect or not aspect.strip():
        return "Please provide both a review and an aspect."
    try:
        res = inferencer.predict(text, aspect)
        sentiment = res['sentiment']
        conf = res['confidence']
        color = SENTIMENT_COLORS.get(sentiment, "gray")
        return f"""
        <div style="background-color: #1e2229; padding: 20px; border-radius: 10px; border-left: 5px solid {color}; color: white;">
            <h4>Target Aspect: <span style="color: #4facfe;">{html.escape(aspect)}</span></h4>
            <h2>Sentiment: <span style="color: {color};">{sentiment}</span> ({conf}%)</h2>
        </div>
        """
    except Exception as e:
        return f"Error: {html.escape(str(e))}"


def _aspect_bar(results_df, sentiment, title):
    counts = results_df[results_df['sentiment'] == sentiment]['aspect'].str.lower().value_counts().head(8)
    if counts.empty:
        return None
    fig = px.bar(x=counts.values, y=counts.index, orientation='h', title=title,
                 labels={'x': 'Mentions', 'y': 'Aspect'}, color_discrete_sequence=[SENTIMENT_COLORS[sentiment]])
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    return fig


@spaces.GPU(duration=120)
def analyze_batch(file_obj):
    if not inferencer:
        return pd.DataFrame([{"Error": "Model failed to load"}]), None, None, None
    if file_obj is None:
        return pd.DataFrame(), None, None, None
    try:
        file_path = file_obj if isinstance(file_obj, str) else file_obj.name
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        if 'text' not in df.columns:
            return pd.DataFrame([{"Error": "CSV must contain a 'text' column"}]), None, None, None

        if 'aspect' in df.columns:
            results_df = inferencer.batch_predict(df)
        elif pipeline is not None:
            texts = df['text'].dropna().astype(str)
            texts = texts[texts.str.strip() != ''].tolist()
            results_df = pd.DataFrame([
                {"text": r["text"], "aspect": a["aspect"], "sentiment": a["sentiment"], "confidence": a["confidence"]}
                for r in pipeline.analyze_batch(texts) for a in r["aspects"]
            ], columns=["text", "aspect", "sentiment", "confidence"])
        else:
            return pd.DataFrame([{"Error": "CSV needs an 'aspect' column (aspect extraction model not loaded)"}]), None, None, None

        if results_df.empty:
            return pd.DataFrame([{"Error": "No aspects found in the uploaded reviews"}]), None, None, None

        sentiment_counts = results_df['sentiment'].value_counts()
        pie = px.pie(names=sentiment_counts.index, values=sentiment_counts.values, title="Overall Sentiment Distribution",
                     color=sentiment_counts.index, color_discrete_map=SENTIMENT_COLORS, hole=0.4)
        pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))

        return (results_df, pie, _aspect_bar(results_df, 'Positive', "Most Praised Aspects"),
                _aspect_bar(results_df, 'Negative', "Most Criticised Aspects"))
    except Exception as e:
        return pd.DataFrame([{"Error": str(e)}]), None, None, None


css = """
body, .gradio-container { background-color: #0f1115 !important; color: white !important; font-family: 'Inter', sans-serif !important; }
h1, h2, h3 { color: #4facfe !important; }
"""

with gr.Blocks(title="Customer Feedback Intelligence") as demo:
    gr.Markdown("# 🧠 Customer Feedback Intelligence")
    gr.Markdown("### End-to-end Aspect-Based Sentiment Analysis: find *what* customers talk about and *how* they feel about each thing.")

    with gr.Tabs():
        with gr.TabItem("🔎 Full Review Analysis"):
            gr.Markdown("Paste a review. The extraction model finds every aspect term, then the sentiment model classifies each one.")
            review_input = gr.Textbox(label="Customer Review", lines=4,
                                      value="The battery life is incredible but the screen is too dim and the keyboard feels cheap.")
            review_btn = gr.Button("Analyze Review", variant="primary")
            highlighted = gr.HighlightedText(label="Detected Aspects", color_map=SENTIMENT_COLORS, show_legend=True)
            aspect_table = gr.Dataframe(label="Aspect Breakdown", value=EMPTY_ASPECTS)
            review_btn.click(fn=analyze_review, inputs=review_input, outputs=[highlighted, aspect_table])
            gr.Examples(
                examples=[
                    ["The pasta was delicious, but the waiter was rude and we waited an hour for the bill."],
                    ["Setup was easy and the trackpad is responsive, though the fan gets loud under load."],
                    ["Great atmosphere and friendly staff, but the prices are a bit steep for the portion sizes."],
                ],
                inputs=review_input,
            )

        with gr.TabItem("💬 Target a Specific Aspect"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(label="Customer Review", lines=5, value="The battery life is incredible but the screen is too dim.")
                with gr.Column(scale=1):
                    aspect_input = gr.Textbox(label="Aspect to Analyze", value="screen")
                    analyze_btn = gr.Button("Predict Sentiment", variant="primary")
            output_html = gr.HTML("### Prediction Results will appear here.")
            analyze_btn.click(fn=analyze_single, inputs=[text_input, aspect_input], outputs=output_html)

        with gr.TabItem("📊 Batch Analytics"):
            gr.Markdown("Upload a CSV with a `text` column. Aspects are extracted automatically; "
                        "add an `aspect` column instead if you want to score specific aspects.")
            file_upload = gr.File(label="Upload CSV", file_types=['.csv'])
            batch_btn = gr.Button("Run Batch Inference", variant="primary")
            pie_chart = gr.Plot(label="Sentiment Distribution")
            with gr.Row():
                pos_chart = gr.Plot(label="Most Praised Aspects")
                neg_chart = gr.Plot(label="Most Criticised Aspects")
            data_output = gr.Dataframe(label="Raw Results")
            batch_btn.click(fn=analyze_batch, inputs=file_upload, outputs=[data_output, pie_chart, pos_chart, neg_chart])

        with gr.TabItem("📈 Model Benchmarks"):
            if os.path.exists(RESULTS_MD):
                with open(RESULTS_MD, encoding="utf-8") as f:
                    gr.Markdown(f.read())
            else:
                gr.Markdown("Benchmark report not found.")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=css)
