import streamlit as st
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Infant Cry Analysis System", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #4A90E2; color: white; }
    h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    .status-box { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; }
    </style>
    """, unsafe_allow_allowed=True)

# --- 2. ASSET LOADING SECTION ---
# This is where we load your AI model and the label text (Hunger, Pain, etc.)
@st.cache_resource
def load_assets():
    try:
        # 'compile=False' is used to bypass version mismatch errors during loading
        model = load_model('infant_cry_classification_model.h5', compile=False)
        with open('label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, encoder
    except Exception as e:
        st.error(f"Error loading assets: {e}. Check if files exist in GitHub.")
        return None, None

model, encoder = load_assets()

# --- 3. SIDEBAR ---
st.sidebar.title("System Info")
st.sidebar.markdown("""
**Core Technology:**
- Feature Extraction: MFCC
- Model: CNN / Deep Learning
- Sampling Rate: 16kHz
""")

# --- 4. MAIN INTERFACE ---
st.title("👶 Smart Infant Cry Analysis")
st.write("Upload a .wav file to bridge the communication gap.")

if model is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Audio")
        uploaded_file = st.file_uploader("Select Audio File", type=["wav"])
        
        if uploaded_file:
            # Load audio for playback
            st.audio(uploaded_file)
            
            # --- VISUALIZATION (Real-time Spectrogram logic) ---
            audio, sr = librosa.load(uploaded_file, sr=16000)
            
            st.write("### Acoustic Spectrogram")
            fig, ax = plt.subplots(figsize=(10, 4))
            S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
            ax.set_title('Mel-Frequency Spectrogram')
            st.pyplot(fig)

    with col2:
        st.subheader("Diagnostic Results")
        if uploaded_file:
            if st.button("Run Diagnostic Analysis"):
                # --- INFERENCE SECTION ---
                # 1. Feature Extraction (Matching your training logic)
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                features = np.mean(mfccs.T, axis=0).reshape(1, -1)
                
                # 2. Model Prediction
                with st.spinner('Analyzing acoustic patterns...'):
                    prediction = model.predict(features, verbose=0)
                    label_index = np.argmax(prediction)
                    label = encoder.inverse_transform([label_index])[0]
                    confidence = np.max(prediction) * 100
                
                # 3. Output Logic (The Dashboard Alert System)
                label_clean = label.replace('_', ' ').upper()
                
                if "PAIN" in label_clean:
                    st.error(f"🚨 **ALERT: {label_clean} DETECTED**")
                    st.toast("Immediate attention required!", icon='🚨')
                elif "HUNGER" in label_clean:
                    st.warning(f"🍼 **STATUS: {label_clean}**")
                else:
                    st.info(f"ℹ️ **STATUS: {label_clean}**")
                
                st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
                st.progress(int(confidence))
                
                # Metadata
                st.write("---")
                st.caption("Note: This is an AI-assisted tool and should not replace professional medical advice.")
else:
    st.warning("Please ensure 'infant_cry_classification_model.h5' and 'label_encoder.pkl' are in your GitHub repository.")
