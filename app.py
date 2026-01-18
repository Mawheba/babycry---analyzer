import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# 1. Page Configuration for a professional look
st.set_page_config(page_title="Infant Cry Analysis System", layout="wide")

# Custom CSS for elegance
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #4A90E2; color: white; }
    .reportview-container .main .block-container{ padding-top: 2rem; }
    h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    </style>
    """, unsafe_allow_allowed=True)

# 2. Asset Loading
@st.cache_resource
def load_assets():
    model = load_model('infant_cry_classification_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_assets()

# 3. Sidebar Information
st.sidebar.title("System Information")
st.sidebar.info("This system utilizes a Deep Neural Network to analyze acoustic features (MFCCs) of infant vocalizations.")

# 4. Main Panel
st.title("Infant Cry Analysis System")
st.write("Upload an audio recording in WAV format to determine the classification.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Select Audio File", type=["wav"])
    if uploaded_file:
        st.audio(uploaded_file)

with col2:
    if uploaded_file:
        if st.button("Run Diagnostic Analysis"):
            # Processing
            audio, sr = librosa.load(uploaded_file, sr=16000)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            features = np.mean(mfccs.T, axis=0).reshape(1, -1)
            
            # Inference
            prediction = model.predict(features, verbose=0)
            label_index = np.argmax(prediction)
            label = encoder.inverse_transform([label_index])[0]
            confidence = np.max(prediction) * 100
            
            # Elegant Output
            st.subheader("Analysis Results")
            st.write(f"Classification: **{label.replace('_', ' ').upper()}**")
            st.write(f"Confidence: **{confidence:.2f}%**")
            st.progress(int(confidence))
