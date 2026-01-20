import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# 1. Page Configuration
st.set_page_config(page_title="Baby Cry Analyzer", page_icon="👶")

st.title("👶 Infant Cry Classification")
st.write("Analyze baby cries using Deep Learning.")

# 2. Safe Model Loading
@st.cache_resource
def load_assets():
    try:
        model = load_model('infant_cry_classification_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, encoder
    except Exception as e:
        return None, None

model, encoder = load_assets()

# 3. User Interface (This will now appear!)
if model is None:
    st.warning("AI Engine is still initializing in the background. Please wait 2 minutes and refresh.")
else:
    st.success("AI Brain Ready!")
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        if st.button("Analyze Cry"):
            with st.spinner("Extracting features..."):
                # Audio Processing
                audio, sr = librosa.load(uploaded_file, res_type='kaiser_fast')
                mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
                
                # Prediction
                prediction = model.predict(mfccs.reshape(1, -1))
                label = encoder.inverse_transform([np.argmax(prediction)])[0]
                
                st.header(f"Result: {label}")
