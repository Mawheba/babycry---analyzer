import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Set Page Title
st.set_config(page_title="Baby Cry Analyzer", page_icon="👶")

st.title("👶 Infant Cry Classification System")
st.markdown("---")

# --- 1. Load the Model and Encoder ---
@st.cache_resource
def load_assets():
    try:
        # Load the Neural Network
        model = load_model('infant_cry_classification_model.h5')
        # Load the Label Translator
        with open('label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, encoder
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

with st.spinner('Loading AI Model... Please wait.'):
    model, encoder = load_assets()

# --- 2. Feature Extraction Function ---
def extract_features(audio_file):
    # This matches the 40 MFCCs used in your training
    audio, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features.reshape(1, -1)

# --- 3. User Interface ---
if model is not None:
    st.write("### Step 1: Upload a Cry Sound")
    uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button('Analyze This Cry'):
            with st.spinner('Analyzing acoustic patterns...'):
                # Process and Predict
                features = extract_features(uploaded_file)
                prediction_logits = model.predict(features)
                prediction_class = np.argmax(prediction_logits, axis=1)
                
                # Decode and Show Result
                result = encoder.inverse_transform(prediction_class)[0]
                
                st.markdown("---")
                st.success(f"## Final Prediction: {result}")
                st.info("This classification is based on the acoustic fingerprint of the audio.")

st.markdown("---")
st.caption("Developed for Infant Cry Classification Project - Powered by TensorFlow & Librosa")
