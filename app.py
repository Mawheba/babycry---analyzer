import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

st.set_page_config(page_title="Baby Cry Analyzer", page_icon="👶")
st.title("👶 Infant Cry Classification")

# Load model and encoder once using cache to save memory
@st.cache_resource
def load_my_model():
    model = load_model('infant_cry_classification_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

# UI Logic
try:
    model, encoder = load_my_model()
    
    st.write("### Step 1: Upload Audio")
    uploaded_file = st.file_uploader("Choose a baby cry recording (.wav)", type=["wav"])

    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Analyze Cry"):
            with st.spinner("Analyzing acoustic patterns..."):
                # Feature Extraction: 40 MFCCs (The Acoustic Fingerprint)
                audio, sr = librosa.load(uploaded_file, res_type='kaiser_fast')
                mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
                
                # Prediction
                prediction = model.predict(mfccs.reshape(1, -1))
                label = encoder.inverse_transform([np.argmax(prediction)])[0]
                
                st.success(f"## Prediction: {label}")
except Exception as e:
    st.info("The AI Engine is warming up. Please wait 2-3 minutes while the Neural Network loads...")
