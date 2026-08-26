import gradio as gr
import pandas as pd
import plotly.express as px
import sys
import os

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.inference.infer import ABSA_Inferencer

print("Loading DeBERTa-v3 Model (this might take a minute on startup)...")
try:
    inferencer = ABSA_Inferencer()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    inferencer = None

def analyze_single(text, aspect):
    if not inferencer:
        return "Error: Model failed to load."
    if not text or not aspect:
        return "Please provide both a review and an aspect."
    try:
        res = inferencer.predict(text, aspect)
        sentiment = res['sentiment']
        conf = res['confidence']
        
        color = "green" if sentiment == "Positive" else "red" if sentiment == "Negative" else "gray"
        
        html_out = f"""
        <div style="background-color: #1e2229; padding: 20px; border-radius: 10px; border-left: 5px solid {color}; color: white;">
            <h4>Target Aspect: <span style="color: #4facfe;">{aspect}</span></h4>
            <h2>Sentiment: <span style="color: {color};">{sentiment}</span> ({conf}%)</h2>
        </div>
        """
        return html_out
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_batch(file_obj):
    if not inferencer:
        return pd.DataFrame([{"Error": "Model failed to load"}]), None, None
    if file_obj is None:
        return pd.DataFrame(), None, None
    try:
        df = pd.read_csv(file_obj.name)
        if 'text' not in df.columns or 'aspect' not in df.columns:
            return pd.DataFrame([{"Error": "CSV must contain 'text' and 'aspect' columns"}]), None, None
            
        results_df = inferencer.batch_predict(df)
        
        sentiment_counts = results_df['sentiment'].value_counts().to_dict()
        top_positive = results_df[results_df['sentiment'] == 'Positive']['aspect'].value_counts().head(5).to_dict()
        
        # Pie chart
        fig1 = px.pie(
            names=list(sentiment_counts.keys()),
            values=list(sentiment_counts.values()),
            title="Overall Sentiment Distribution",
            color_discrete_sequence=['#4caf50', '#f44336', '#9e9e9e']
        )
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        
        # Bar chart
        pos_df = pd.DataFrame(list(top_positive.items()), columns=['Aspect', 'Count'])
        fig2 = None
        if not pos_df.empty:
            fig2 = px.bar(pos_df, x='Count', y='Aspect', orientation='h', title="Top Positive Aspects", color_discrete_sequence=['#4caf50'])
            fig2.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            
        return results_df, fig1, fig2
    except Exception as e:
        return pd.DataFrame([{"Error": str(e)}]), None, None

css = """
body, .gradio-container { background-color: #0f1115 !important; color: white !important; font-family: 'Inter', sans-serif !important; }
h1, h2, h3 { color: #4facfe !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("# 🧠 Customer Feedback Intelligence")
    gr.Markdown("### Aspect-Based Sentiment Analysis (ABSA) powered by Transformers")
    gr.Markdown("Extract precise sentiments directed at specific aspects of a product or service.")
    
    with gr.Tabs():
        with gr.TabItem("💬 Single Review Analysis"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(label="Customer Review", lines=5, value="The battery life is incredible but the screen is too dim.")
                with gr.Column(scale=1):
                    aspect_input = gr.Textbox(label="Aspect to Analyze", value="battery life")
                    analyze_btn = gr.Button("Predict Sentiment", variant="primary")
            
            output_html = gr.HTML("### Prediction Results will appear here.")
            analyze_btn.click(fn=analyze_single, inputs=[text_input, aspect_input], outputs=output_html)
            
        with gr.TabItem("📊 Batch Analytics"):
            gr.Markdown("Upload a CSV with `text` and `aspect` columns to process hundreds of reviews instantly.")
            file_upload = gr.File(label="Upload CSV", file_types=['.csv'])
            batch_btn = gr.Button("Run Batch Inference", variant="primary")
            
            with gr.Row():
                pie_chart = gr.Plot(label="Sentiment Distribution")
                bar_chart = gr.Plot(label="Top Positive Aspects")
                
            data_output = gr.Dataframe(label="Raw Results")
            
            batch_btn.click(fn=analyze_batch, inputs=file_upload, outputs=[data_output, pie_chart, bar_chart])

if __name__ == "__main__":
    demo.launch()
