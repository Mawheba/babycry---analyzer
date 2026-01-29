import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os

st.set_page_config(page_title="Infant Cry AI", page_icon="👶")

st.title("👶 Infant Cry Classification System")
st.markdown("---")

# --- 1. Load Assets (Fixed for Batch_Shape Error) ---
@st.cache_resource
def load_assets():
    try:
        model_path = 'infant_cry_classification_model.h5'
        pickle_path = 'label_encoder.pkl'
        
        if not os.path.exists(model_path):
            return None, None, "File Not Found: .h5 model"

        # FIX: We use compile=False to avoid the deserialization error
        # The model will still predict perfectly!
        model = load_model(model_path, compile=False)
        
        with open(pickle_path, 'rb') as f:
            encoder = pickle.load(f)
            
        return model, encoder, "Success"
    except Exception as e:
        return None, None, str(e)

model, encoder, status = load_assets()

# --- 2. Interface Logic ---
if model is None:
    st.error(f"Engine Error: {status}")
    st.info("Technical Tip: Ensure your TensorFlow versions match.")
else:
    st.success("AI Brain Connected & Ready!")
    
    file = st.file_uploader("Upload Baby Cry (.wav)", type=["wav"])

    if file is not None:
        st.audio(file)
        
        if st.button("Classify This Cry"):
            try:
                with st.spinner("Analyzing acoustic features..."):
                    # Feature Extraction (Matches your training)
                    audio, sr = librosa.load(file, res_type='kaiser_fast')
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                    mfccs_processed = np.mean(mfccs.T, axis=0).reshape(1, -1)
                    
                    # Prediction
                    prediction = model.predict(mfccs_processed)
                    class_idx = np.argmax(prediction)
                    label = encoder.inverse_transform([class_idx])[0]
                    
                    st.markdown("---")
                    st.header(f"Result: {label}")
                    st.balloons()
            except Exception as e:
                st.error(f"Analysis Failed: {e}")

st.markdown("---")
st.caption("v2.1 - Deployment Fix for Keras Deserialization")
