import streamlit as st
from PIL import Image
import time
import numpy as np
import pandas as pd
import base64
import joblib  # For model loading
import os

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

# Load trained model (assumes model.pkl is a scikit-learn pipeline)
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        st.error("Model file 'model.pkl' not found. Please train your model and place 'model.pkl' in the project directory.")
        st.stop()
    return joblib.load("model.pkl")
model = load_model()

# Animated header
st.markdown("""
    <style>
    .animated-title {
        font-size: 2.5em;
        font-weight: bold;
        background: linear-gradient(90deg, #ff5858, #f09819, #43cea2, #185a9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-move 3s infinite alternate;
    }
    @keyframes gradient-move {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }
    </style>
    <div class="animated-title">📰 Fake News Detection App</div>
""", unsafe_allow_html=True)

st.write("")
st.markdown("""
Welcome to the **Fake News Detection** app! Upload a news article or type your own, and let our AI-powered model tell you if it's real or fake. Enjoy interactive visuals and a modern, animated interface.
""")

# Animation: Loading bar
with st.spinner('Loading model and preparing interface...'):
    time.sleep(1.5)

# Example animation: Bouncing emoji
st.markdown("""
<div style="text-align:center; font-size: 2em;">
    <span style="display:inline-block; animation:bounce 1s infinite alternate;">📰</span>
</div>
<style>
@keyframes bounce {
  0% { transform: translateY(0); }
  100% { transform: translateY(-20px); }
}
</style>
""", unsafe_allow_html=True)

# User input
st.subheader("Enter News Article")
user_input = st.text_area("Paste the news article text here:", height=200)

# File upload
uploaded_file = st.file_uploader("Or upload a CSV file with a 'text' column:", type=["csv"])

# Placeholder for prediction result
result_placeholder = st.empty()

# Prediction function using the loaded model
def predict_fake_news(text):
    with st.spinner('Analyzing...'):
        time.sleep(1)  # Optional: keep a short delay for UI effect
    pred = model.predict([text])[0]
    if hasattr(model, "predict_proba"):
        conf = np.max(model.predict_proba([text]))
    else:
        conf = 0.99  # fallback if no probability
    label = "Fake" if pred == 1 or str(pred).lower() == "fake" else "Real"
    return label, conf

# Predict button
if st.button("Detect News Authenticity", use_container_width=True):
    if user_input.strip():
        label, confidence = predict_fake_news(user_input)
        if label == "Fake":
            color = "#ff5858"
            emoji = "🚨"
        else:
            color = "#43cea2"
            emoji = "✅"
        result_placeholder.markdown(f"""
            <div style='padding: 1em; border-radius: 10px; background: {color}; color: white; text-align: center; font-size: 1.5em; animation: fadeIn 1s;'>
                {emoji} <b>{label} News</b> <br> Confidence: {confidence:.2%}
            </div>
            <style>@keyframes fadeIn {{0%{{opacity:0;}}100%{{opacity:1;}}}}</style>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please enter some text or upload a file.")

# If file uploaded, show predictions for each row
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'text' in df.columns:
        st.write("## Batch Prediction Results")
        progress = st.progress(0)
        results = []
        for i, row in df.iterrows():
            label, confidence = predict_fake_news(row['text'])
            results.append({"Text": row['text'][:50]+"...", "Prediction": label, "Confidence": f"{confidence:.2%}"})
            progress.progress((i+1)/len(df))
        st.dataframe(pd.DataFrame(results))
    else:
        st.error("CSV must contain a 'text' column.")

# Footer with animation
st.markdown("""
---
<div style='text-align:center; font-size:1.1em;'>
    Made with ❤️ using <b>Streamlit</b> &nbsp;|&nbsp; <span style='animation: pulse 1s infinite alternate;'>🤖</span>
</div>
<style>
@keyframes pulse {
  0% { opacity: 0.5; }
  100% { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)