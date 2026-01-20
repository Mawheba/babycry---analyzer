import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os

# 1. Initialize Page
st.set_page_config(page_title="Baby Cry Analyzer", page_icon="👶")
st.title("👶 Infant Cry Classification")
st.write("Upload a baby cry recording (.wav) to see the prediction.")

# 2. Optimized Loading (Stays in memory)
@st.cache_resource
def load_assets():
    try:
        # Verify files exist in GitHub
        if not os.path.exists('infant_cry_classification_model.h5'):
            return None, "Missing file: infant_cry_classification_model.h5"
        
        model = load_model('infant_cry_classification_model.h5')
        
        with open('label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
            
        return (model, encoder), "Success"
    except Exception as e:
        return None, str(e)

# Load the AI components
assets, status_msg = load_assets()

# 3. User Interface
if assets is None:
    st.error(f"System not ready: {status_msg}")
    st.info("Ensure your .h5 and .pkl files are in the main folder of your GitHub repository.")
else:
    model, encoder = assets
    st.success("AI Brain Connected!")
    
    # --- INPUT SECTION ---
    uploaded_file = st.file_uploader("Select a .wav file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        # --- OUTPUT SECTION ---
        if st.button("Analyze Cry Now"):
            with st.spinner("Decoding acoustic patterns..."):
                # Processing: Extracting 40 MFCCs
                audio, sr = librosa.load(uploaded_file, res_type='kaiser_fast')
                mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
                
                # Model Prediction
                prediction = model.predict(mfccs.reshape(1, -1))
                result = encoder.inverse_transform([np.argmax(prediction)])[0]
                
                st.markdown("---")
                st.success(f"## Prediction: {result}")
