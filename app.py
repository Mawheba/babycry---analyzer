import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os

# 1. Page Setup
st.set_page_config(page_title="Baby Cry Analyzer", page_icon="👶")

st.title("👶 Infant Cry Classification")
st.write("Upload a .wav file to classify the baby's cry.")

# 2. Optimized Model Loading
@st.cache_resource
def load_assets():
    try:
        # Check if files exist to prevent crashes
        if not os.path.exists('infant_cry_classification_model.h5'):
            return None, "Model file (.h5) not found in GitHub!"
        
        model = load_model('infant_cry_classification_model.h5')
        
        with open('label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
            
        return (model, encoder), "Success"
    except Exception as e:
        return None, str(e)

# Initialize the AI
assets, status_msg = load_assets()

# 3. User Interface Logic
if assets is None:
    st.error(f"System Error: {status_msg}")
    st.info("Make sure 'infant_cry_classification_model.h5' and 'label_encoder.pkl' are in your GitHub folder.")
else:
    model, encoder = assets
    st.success("AI Brain Connected!")
    
    # --- INPUT ---
    uploaded_file = st.file_uploader("Choose a baby cry recording (.wav)", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        # --- OUTPUT ---
        if st.button("Classify Cry"):
            with st.spinner("Analyzing acoustic patterns..."):
                # Feature Extraction (40 MFCCs)
                audio, sr = librosa.load(uploaded_file, res_type='kaiser_fast')
                mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
                
                # Prediction
                prediction = model.predict(mfccs.reshape(1, -1))
                result = encoder.inverse_transform([np.argmax(prediction)])[0]
                
                st.markdown(f"## Prediction: **{result}**")
