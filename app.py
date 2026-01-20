import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

st.set_page_config(page_title="Baby Cry Analyzer", page_icon="👶")

st.title("👶 Infant Cry Classification")
st.write("Ready for analysis. Upload a .wav file below.")

# 1. Faster Model Loading
@st.cache_resource
def load_my_assets():
    try:
        model = load_model('infant_cry_classification_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, encoder
    except Exception as e:
        return None, str(e)

# 2. Status Check
model, encoder = load_my_assets()

if model is None:
    st.error(f"Waiting for AI Engine: {encoder}")
    st.info("The server is still connecting to the model files. Please wait 2 minutes.")
else:
    st.success("AI Brain Connected!")
    
    # 3. Input Section
    uploaded_file = st.file_uploader("Select a baby cry recording (.wav)", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        if st.button("Classify Cry"):
            with st.spinner("Extracting acoustic fingerprints..."):
                # Processing
                audio, sr = librosa.load(uploaded_file, res_type='kaiser_fast')
                mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
                
                # Prediction
                prediction = model.predict(mfccs.reshape(1, -1))
                result = encoder.inverse_transform([np.argmax(prediction)])[0]
                
                st.markdown(f"### Prediction: **{result}**")
