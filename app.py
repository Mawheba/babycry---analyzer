import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

st.set_page_config(page_title="Baby Cry Analyzer", page_icon="👶")

# Corrected Markdown section (fixed the 'unsafe_allow_html' typo)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("👶 Infant Cry Classification")

# Load model and encoder using caching
@st.cache_resource
def load_my_model():
    model = load_model('infant_cry_classification_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

try:
    model, encoder = load_my_model()
    
    st.write("### Step 1: Upload a Cry Recording")
    uploaded_file = st.file_uploader("Select a .wav file", type=["wav"])

    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Analyze Cry Patterns"):
            with st.spinner("Processing audio fingerprints..."):
                # Feature Extraction (40 MFCCs)
                audio, sr = librosa.load(uploaded_file, res_type='kaiser_fast')
                mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
                
                # Model Prediction
                prediction = model.predict(mfccs.reshape(1, -1))
                label = encoder.inverse_transform([np.argmax(prediction)])[0]
                
                st.success(f"## Result: {label}")
except Exception as e:
    st.info("The AI Engine is warming up. Please wait 1-2 minutes...")
