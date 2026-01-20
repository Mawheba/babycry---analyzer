import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

st.set_page_config(page_title="Baby Cry Analyzer", page_icon="👶")
st.title("👶 Infant Cry Classification")

# Load model and encoder once
@st.cache_resource
def load_my_model():
    model = load_model('infant_cry_classification_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

# UI Logic
try:
    model, encoder = load_my_model()
    
    uploaded_file = st.file_uploader("Upload Baby Cry (.wav)", type=["wav"])

    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Analyze Cry"):
            with st.spinner("Analyzing..."):
                # Processing: 40 MFCCs (The Acoustic Fingerprint)
                audio, sr = librosa.load(uploaded_file, res_type='kaiser_fast')
                mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
                
                # Prediction
                prediction = model.predict(mfccs.reshape(1, -1))
                label = encoder.inverse_transform([np.argmax(prediction)])[0]
                
                st.success(f"Result: {label}")
except Exception as e:
    st.info("The AI Engine is warming up. Please wait 1-2 minutes...")
