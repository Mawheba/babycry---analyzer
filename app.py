import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os

# Page Config
st.set_page_config(page_title="Infant Cry Analyzer", page_icon="👶")

st.title("👶 Infant Cry Classification System")
st.write("Upload a baby cry sound file (.wav) to analyze the cause.")

# --- Background AI Loading ---
@st.cache_resource
def load_ai_model():
    try:
        # Verify files exist
        model_path = 'infant_cry_classification_model.h5'
        pickle_path = 'label_encoder.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(pickle_path):
            return None, "Model or Encoder file missing in GitHub repository."
        
        # Load the Neural Network
        model = load_model(model_path)
        # Load the Label Translator
        with open(pickle_path, 'rb') as f:
            encoder = pickle.load(f)
        return (model, encoder), "Success"
    except Exception as e:
        return None, str(e)

# Start Loading
data, message = load_ai_model()

# --- User Interface Logic ---
if data is None:
    st.error(f"Initialization Error: {message}")
    st.info("Please ensure your .h5 and .pkl files are uploaded to GitHub.")
else:
    model, encoder = data
    st.success("AI Brain Connected!")

    # Step 1: Input
    uploaded_file = st.file_uploader("Upload Baby Cry (.wav)", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        # Step 2: Process & Output
        if st.button("Analyze Cry Patterns"):
            with st.spinner("Extracting acoustic fingerprints..."):
                # 1. Load audio
                audio, sr = librosa.load(uploaded_file, res_type='kaiser_fast')
                # 2. Extract 40 MFCCs (The Feature Vector)
                mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
                # 3. Reshape for the model
                features = mfccs.reshape(1, -1)
                
                # 4. Predict
                prediction = model.predict(features)
                class_index = np.argmax(prediction)
                result = encoder.inverse_transform([class_index])[0]
                
                # Show Result
                st.markdown("---")
                st.header(f"Result: {result}")
                st.balloons()

st.markdown("---")
st.caption("Technical Details: Uses a Deep Neural Network with 40-dimensional MFCC feature extraction.")
