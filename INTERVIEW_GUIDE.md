# Customer Feedback Intelligence — Interview Guide

This guide contains the key talking points you need to confidently explain this project in a machine learning or data science interview.

## 1. Project Overview & Business Value
**What to say:** "I built an Aspect-Based Sentiment Analysis (ABSA) system. Standard sentiment analysis just tells you a review is 'Negative'. My system tells you exactly *what* the customer is unhappy about. For example, 'The battery life is incredible but the screen is too dim' is processed to extract 'battery' (Positive) and 'screen' (Negative). This gives product managers actionable intelligence instead of just an overall score."

## 2. The Data Pipeline (Phase 1)
**What to say:** "I started with the SemEval-2014 Task 4 dataset, which is the academic gold standard for this problem. It was provided in raw XML. I built a robust parsing pipeline to extract the sentences, the aspect terms, and the polarity. I used `spaCy` for tokenization and cleaning because it's much faster and more production-ready than NLTK."

## 3. The Baseline (Phase 2)
**What to say:** "I didn't just jump to deep learning. A good engineer establishes a baseline. I used TF-IDF vectorization and a Linear SVM. It achieved about ~71% accuracy, but struggled with complex syntax like double negatives and long-range dependencies."

## 4. Deep Learning & The Attention Mechanism (Phase 3 & 4)
**What to say:** "To improve performance, I built a custom PyTorch architecture from scratch. I used a Bidirectional LSTM (BiLSTM) so the network could read the sentence forwards and backwards. More importantly, I mathematically implemented a **Custom Attention Layer**. 
By feeding both the sentence and the target aspect into the attention mechanism, the model dynamically learns to *ignore* irrelevant words and focus entirely on the adjectives modifying the specific aspect. This pushed accuracy to ~74.4%."

## 5. State of the Art Transformers (Phase 5)
**What to say:** "Finally, I implemented the current State-of-the-Art approach. I used HuggingFace to fine-tune a pre-trained Transformer (like BERT/DeBERTa). Because Transformers use multi-headed Self-Attention and are pre-trained on billions of words, the model inherently understands English grammar. To pass both the sentence and the aspect to the model, I used the `[CLS] text [SEP] aspect [SEP]` architecture. This blew the previous models away, achieving ~82% accuracy."

## 6. Error Analysis & Interpretability (Phase 6)
**What to say:** "I didn't treat the model like a black box. I wrote scripts to extract the attention weights ($\alpha$) directly from my PyTorch network and plotted them as heatmaps. I could visually prove to stakeholders that when predicting the sentiment of 'service', the model was correctly assigning the highest mathematical weight to the word 'terrible'. I also identified the model's main failure modes: implicit sarcasm and complex double negatives."

## 7. The Production App (Phases 8-11)
**What to say:** "A model is useless if it just sits in a Jupyter Notebook. I engineered a production inference pipeline using object-oriented Python. I deployed this pipeline behind a **FastAPI** REST backend because of its high performance and async capabilities. Finally, I built a premium **Streamlit** dashboard that allows users to type in single reviews or upload bulk CSVs to get dynamic Plotly analytics of their customer feedback."

---
**Good luck! You've built an incredibly impressive, end-to-end ML project!**
