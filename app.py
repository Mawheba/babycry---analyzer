import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Set Page Title
st.set_page_config(page_title="Baby Cry Analyzer", page_icon="👶")

st.title("👶 Infant Cry Classification System")
st.write("Upload a .wav file to analyze the baby's needs.")

# --- 1. Load the Model and Encoder ---
@st.cache_resource
def load_assets():
    # Make sure these filenames match exactly what is in your GitHub
    model = load_model('infant_cry_classification_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

with st.spinner('Initializing AI Engine... Please wait.'):
    model, encoder = load_assets()

# --- 2. Feature Extraction Function ---
def extract_features(audio_path):
    # This matches the 40 MFCCs you used in your Colab training
    audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features.reshape(1, -1)

# --- 3. UI and Prediction ---
uploaded_file = st.file_uploader("Choose a baby cry recording (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button('Analyze Cry'):
        with st.spinner('Analyzing acoustic patterns...'):
            # 1. Extract Features
            features = extract_features(uploaded_file)
            
            # 2. Predict
            prediction_logits = model.predict(features)
            prediction_class = np.argmax(prediction_logits, axis=1)
            
            # 3. Decode Label
            result = encoder.inverse_transform(prediction_class)[0]
            
            # 4. Display Result
            st.success(f"### Prediction: {result}")
            
            # Advice logic (Optional but looks good for presentation)
            if result.lower() == 'hungry':
                st.info("Advice: The baby might need feeding soon.")
            elif result.lower() == 'pain':
                st.warning("Advice: Check for physical discomfort or gas.")
