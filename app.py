import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os

# Page Configuration
st.set_page_config(page_title="Baby Cry Analyzer", page_icon="👶")

st.title("👶 Infant Cry Classification")
st.write("Analyze baby cries using Deep Learning.")

# Model Loading with improved caching and error handling
@st.cache_resource
def load_assets():
    try:
        if not os.path.exists('infant_cry_classification_model.h5'):
            raise FileNotFoundError("Model file 'infant_cry_classification_model.h5' not found. Upload to repo.")
        if not os.path.exists('label_encoder.pkl'):
            raise FileNotFoundError("Label encoder 'label_encoder.pkl' not found. Upload to repo.")
        
        model = load_model('infant_cry_classification_model.h5', compile=False)
        with open('label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, encoder
    except Exception as e:
        st.error(f"Loading failed: {str(e)}. Ensure files are in repo root (use Git LFS for large .h5).")[web:3][web:15]
        return None, None

model, encoder = load_assets()

if model is None:
    st.warning("❌ AI Engine unavailable. Check files and refresh.")
else:
    st.success("✅ AI Brain Ready!")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload baby cry (.wav)", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        
        if st.button("🔍 Analyze Cry", type="primary"):
            with st.spinner("Extracting MFCC features..."):
                # Load and process audio
                audio, sr = librosa.load(uploaded_file, sr=22050, res_type='kaiser_fast')
                
                # Extract 40 MFCCs (mean pooled as in original)
                mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40), axis=1)
                
                # Reshape for model input (1, 40)
                mfccs = mfccs.reshape(1, -1)
                
                # Predict
                prediction = model.predict(mfccs, verbose=0)
                label_idx = np.argmax(prediction, axis=1)[0]
                label = encoder.inverse_transform([label_idx])[0]
                confidence = np.max(prediction) * 100
                
                # Results
                st.markdown(f"### 🎯 **Prediction: {label}** (Confidence: {confidence:.1f}%)")
                st.bar_chart({encoder.classes_[i]: f"{prediction[0][i]*100:.1f}%" for i in range(len(encoder.classes_))})[web:1][web:5]

# Footer
st.markdown("---")
st.caption("Built with ❤️ using MFCC features and Keras. Train on Donate-a-Cry dataset for best results.")[web:1][web:5]
