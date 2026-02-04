import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os

st.set_page_config(page_title="Infant Cry AI", page_icon="👶")
st.title("👶 Infant Cry Classification System")

@st.cache_resource
def load_assets():
    try:
        model_path = 'infant_cry_classification_model.h5'
        pickle_path = 'label_encoder.pkl'
        
        if not os.path.exists(model_path):
            return None, None, "Model file not found in GitHub."

        # THE FINAL BYPASS: safe_mode=False allows Keras 3 to attempt 
        # a legacy load of a Keras 2 file structure.
        model = load_model(model_path, compile=False, safe_mode=False)
        
        with open(pickle_path, 'rb') as f:
            encoder = pickle.load(f)
            
        return model, encoder, "Success"
    except Exception as e:
        return None, None, str(e)

model, encoder, status = load_assets()

if model is None:
    st.error(f"Engine Error: {status}")
    st.info("If you see 'batch_shape', the system is still using Keras 3. See reboot steps below.")
else:
    st.success("AI Brain Connected!")
    file = st.file_uploader("Upload .wav recording", type=["wav"])
    
    if file:
        st.audio(file)
        if st.button("Analyze Cry"):
            with st.spinner("Extracting acoustic features..."):
                # Audio Processing: 40 MFCCs
                audio, sr = librosa.load(file, res_type='kaiser_fast')
                mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0).reshape(1, -1)
                
                # Prediction
                prediction = model.predict(mfccs)
                label = encoder.inverse_transform([np.argmax(prediction)])[0]
                
                st.markdown("---")
                st.header(f"Result: {label}")
                st.balloons()
