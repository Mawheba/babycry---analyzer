import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os

# Set Page Title
st.set_page_config(page_title="Baby Cry Analyzer", page_icon="👶")

st.title("👶 Infant Cry Classification System")
st.markdown("---")

# --- 1. Load the Model and Encoder ---
# We use st.cache_resource so the model stays in memory and doesn't reload every time
@st.cache_resource
def load_assets():
    try:
        # Loading the Brain (Neural Network)
        model = load_model('infant_cry_classification_model.h5')
        # Loading the Translator (Label Encoder)
        with open('label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, encoder
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

with st.spinner('Initializing AI Engine... This may take a minute.'):
    model, encoder = load_assets()

# --- 2. Feature Extraction Function ---
def extract_features(audio_file):
    # This matches the 40 MFCCs from your Colab training
    audio, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features.reshape(1, -1)

# --- 3. User Interface ---
st.write("### 1. Upload Audio")
uploaded_file = st.file_uploader("Choose a baby cry recording (.wav)", type=["wav"])

if uploaded_file is not None:
    # Play the uploaded audio back to the user
    st.audio(uploaded_file, format='audio/wav')
    
    st.write("### 2. Analysis")
    if st.button('Classify This Cry'):
        if model is not None:
            with st.spinner('Extracting acoustic fingerprints...'):
                # Step 1: Extract MFCCs
                features = extract_features(uploaded_file)
                
                # Step 2: Prediction
                prediction_logits = model.predict(features)
                prediction_class = np.argmax(prediction_logits, axis=1)
                
                # Step 3: Decode Label
                result = encoder.inverse_transform(prediction_class)[0]
                
                # Step 4: Display Result
                st.success(f"## Prediction: {result}")
                
                # Add helpful context for the presentation
                st.info(f"The model identified patterns consistent with the '{result}' category.")
        else:
            st.error("Model not loaded. Check if the .h5 file is in the GitHub repository.")

st.markdown("---")
st.caption("Technical Note: This system uses a Deep Neural Network to analyze 40-dimensional MFCC vectors.")
