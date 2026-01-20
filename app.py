import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

st.set_page_config(page_title="Baby Cry Analyzer", page_icon="👶")

st.title("👶 Infant Cry Classification")
st.write("Upload a .wav file to classify the baby's cry.")

# --- 1. Load Model Assets ---
@st.cache_resource
def load_my_assets():
    try:
        # These filenames must match your GitHub exactly
        model = load_model('infant_cry_classification_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, encoder
    except Exception as e:
        return None, str(e)

model, encoder = load_my_assets()

# --- 2. Interface Logic ---
if model is None:
    st.error(f"Waiting for AI Engine to load: {encoder}")
    st.info("Please wait 1-2 minutes for the initial connection.")
else:
    st.success("AI Brain Connected!")
    
    # This is your Input
    uploaded_file = st.file_uploader("Choose a baby cry recording (.wav)", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        # This triggers the Output
        if st.button("Classify Cry"):
            with st.spinner("Analyzing audio patterns..."):
                # Feature Extraction (40 MFCCs)
                audio, sr = librosa.load(uploaded_file, res_type='kaiser_fast')
                mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
                
                # Prediction
                prediction = model.predict(mfccs.reshape(1, -1))
                result = encoder.inverse_transform([np.argmax(prediction)])[0]
                
                # This is your Output
                st.markdown(f"## Prediction: **{result}**")
