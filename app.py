import streamlit as st
import pandas as pd
import io
import plotly.express as px
import sys
import os

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.inference.infer import ABSA_Inferencer

# Configure page settings
st.set_page_config(
    page_title="Customer Feedback Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark mode premium aesthetics
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0f1115;
        color: #ffffff;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #4facfe;
        font-family: 'Inter', sans-serif;
    }
    
    /* Input Boxes */
    .stTextInput > div > div > input {
        background-color: #1e2229;
        color: #ffffff;
        border: 1px solid #333945;
        border-radius: 8px;
    }
    
    /* Upload Box */
    .stFileUploader > div > div {
        background-color: #1e2229;
        border-radius: 8px;
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 1px solid #333945;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("🧠 Customer Feedback Intelligence")
st.markdown("### Aspect-Based Sentiment Analysis (ABSA) powered by Transformers")
st.write("Extract precise sentiments directed at specific aspects of a product or service.")
st.markdown("---")

# Cache the model so it only loads once per session
@st.cache_resource
def load_model():
    return ABSA_Inferencer()

with st.spinner("Loading DeBERTa-v3 Model (this takes a moment on startup)..."):
    try:
        inferencer = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

tab1, tab2 = st.tabs(["💬 Single Review Analysis", "📊 Batch Analytics"])

# ----------------- TAB 1: SINGLE REVIEW -----------------
with tab1:
    st.header("Analyze a Single Review")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        text_input = st.text_area("Customer Review", "The battery life is incredible but the screen is too dim.", height=150)
    with col2:
        aspect_input = st.text_input("Aspect to Analyze", "battery life")
        analyze_btn = st.button("Predict Sentiment", type="primary", use_container_width=True)
        
    if analyze_btn:
        if not text_input or not aspect_input:
            st.warning("Please provide both a review and an aspect.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    res = inferencer.predict(text_input, aspect_input)
                    sentiment = res['sentiment']
                    conf = res['confidence']
                    
                    st.markdown("### Prediction Results")
                    
                    # Color coding
                    color = "green" if sentiment == "Positive" else "red" if sentiment == "Negative" else "gray"
                    
                    st.markdown(f"""
                    <div style="background-color: #1e2229; padding: 20px; border-radius: 10px; border-left: 5px solid {color};">
                        <h4>Target Aspect: <span style="color: #4facfe;">{aspect_input}</span></h4>
                        <h2>Sentiment: <span style="color: {color};">{sentiment}</span> ({conf}%)</h2>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# ----------------- TAB 2: BATCH ANALYTICS -----------------
with tab2:
    st.header("Batch Upload & Analytics")
    st.markdown("Upload a CSV with `text` and `aspect` columns to process hundreds of reviews instantly.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        if st.button("Run Batch Inference", type="primary"):
            with st.spinner("Processing batch locally..."):
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'text' not in df.columns or 'aspect' not in df.columns:
                        st.error("CSV must contain 'text' and 'aspect' columns.")
                    else:
                        results_df = inferencer.batch_predict(df)
                        
                        # Calculate aggregate analytics
                        sentiment_counts = results_df['sentiment'].value_counts().to_dict()
                        top_positive = results_df[results_df['sentiment'] == 'Positive']['aspect'].value_counts().head(5).to_dict()
                        top_negative = results_df[results_df['sentiment'] == 'Negative']['aspect'].value_counts().head(5).to_dict()
                        
                        summary = {
                            "total_analyzed": len(results_df),
                            "sentiment_breakdown": sentiment_counts,
                            "top_positive_aspects": top_positive,
                            "top_negative_aspects": top_negative
                        }
                        
                        st.success(f"Successfully analyzed {summary['total_analyzed']} reviews!")
                        
                        # Visualizations
                        st.markdown("### Executive Summary")
                        c1, c2 = st.columns(2)
                        
                        with c1:
                            # Pie chart for sentiments
                            fig1 = px.pie(
                                names=list(summary['sentiment_breakdown'].keys()),
                                values=list(summary['sentiment_breakdown'].values()),
                                title="Overall Sentiment Distribution",
                                color_discrete_sequence=['#4caf50', '#f44336', '#9e9e9e'],
                                hole=0.4
                            )
                            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                            st.plotly_chart(fig1, use_container_width=True)
                            
                        with c2:
                            # Top positive aspects
                            pos_df = pd.DataFrame(list(summary['top_positive_aspects'].items()), columns=['Aspect', 'Count'])
                            if not pos_df.empty:
                                fig2 = px.bar(pos_df, x='Count', y='Aspect', orientation='h', title="Top Positive Aspects", color_discrete_sequence=['#4caf50'])
                                fig2.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                                st.plotly_chart(fig2, use_container_width=True)

                        st.markdown("### Raw Results")
                        st.dataframe(results_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error during batch processing: {e}")
