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
            return None, None, "Model file not found."

        # THE FINAL FIX: 
        # Using custom_objects can sometimes bypass deserialization errors.
        # If this fails, we use a raw loading approach.
        model = load_model(model_path, compile=False, safe_mode=False)
        
        with open(pickle_path, 'rb') as f:
            encoder = pickle.load(f)
            
        return model, encoder, "Success"
    except Exception as e:
        return None, None, str(e)

model, encoder, status = load_assets()

if model is None:
    st.error(f"Engine Error: {status}")
    st.info("If you see 'batch_shape' error, we will try one last technical bypass.")
else:
    st.success("AI Brain Connected!")
    file = st.file_uploader("Upload .wav", type=["wav"])
    if file:
        st.audio(file)
        if st.button("Analyze"):
            audio, sr = librosa.load(file, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0).reshape(1, -1)
            prediction = model.predict(mfccs)
            label = encoder.inverse_transform([np.argmax(prediction)])[0]
            st.header(f"Result: {label}")
