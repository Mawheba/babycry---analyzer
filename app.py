import streamlit as st
import librosa
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras

# Page configuration
st.set_page_config(
    page_title="Infant Cry Classifier",
    page_icon="👶",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
    }
    .prediction-label {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
    }
    .info-text {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and label encoder with caching
@st.cache_resource
def load_model_and_encoder():
    """Load the pre-trained model and label encoder"""
    try:
        # Load model without compiling to avoid batch_shape errors
        model = tf.keras.models.load_model(
            'infant_cry_classification_model.h5',
            compile=False
        )
        
        # Load label encoder
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        return model, label_encoder
    except Exception as e:
        st.error(f"Error loading model or label encoder: {str(e)}")
        return None, None

def extract_mfcc_features(audio_file):
    """
    Extract MFCC features from audio file
    
    Parameters:
    -----------
    audio_file : UploadedFile
        The uploaded audio file
    
    Returns:
    --------
    numpy.ndarray
        MFCC features of shape (1, 40)
    """
    try:
        # Load audio file using librosa
        audio_data, sample_rate = librosa.load(audio_file, sr=None)
        
        # Extract 40 MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        
        # Calculate mean across time axis to get 1D vector of shape (40,)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # Reshape to (1, 40) for model input
        mfcc_features = mfcc_mean.reshape(1, 40)
        
        return mfcc_features
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def predict_cry_type(model, label_encoder, features):
    """
    Predict the type of infant cry
    
    Parameters:
    -----------
    model : keras.Model
        The loaded Keras model
    label_encoder : LabelEncoder
        The loaded label encoder
    features : numpy.ndarray
        MFCC features of shape (1, 40)
    
    Returns:
    --------
    str
        The predicted cry type
    """
    try:
        # Make prediction
        prediction = model.predict(features, verbose=0)
        
        # Get predicted class index
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        
        # Decode the label
        predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return predicted_label
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Main application
def main():
    # Header
    st.markdown('<p class="main-header">👶 Infant Cry Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Upload an audio file to classify the type of infant cry</p>', unsafe_allow_html=True)
    
    # Load model and encoder
    model, label_encoder = load_model_and_encoder()
    
    if model is None or label_encoder is None:
        st.error("⚠️ Could not load model or label encoder. Please ensure the files exist in the same directory.")
        st.info("Required files: `infant_cry_classification_model.h5` and `label_encoder.pkl`")
        return
    
    # File uploader
    st.markdown("### 📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose a WAV file",
        type=['wav'],
        help="Upload a .wav audio file of an infant crying"
    )
    
    if uploaded_file is not None:
        # Display audio player
        st.markdown("### 🔊 Audio Playback")
        st.audio(uploaded_file, format='audio/wav')
        
        # Add a divider
        st.markdown("---")
        
        # Process button
        if st.button("🔍 Classify Cry", type="primary", use_container_width=True):
            with st.spinner("Analyzing audio..."):
                # Extract features
                features = extract_mfcc_features(uploaded_file)
                
                if features is not None:
                    # Make prediction
                    prediction = predict_cry_type(model, label_encoder, features)
                    
                    if prediction is not None:
                        # Display prediction
                        st.markdown("### 🎯 Classification Result")
                        st.markdown(f"""
                            <div class="prediction-box">
                                <p style="font-size: 1.2rem; color: #666; margin-bottom: 0.5rem;">
                                    The infant cry is classified as:
                                </p>
                                <p class="prediction-label">{prediction}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Success animation
                        st.balloons()
                        st.success("✅ Classification completed successfully!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #888; font-size: 0.9rem;">
            <p>💡 This application uses deep learning to classify infant cries</p>
            <p>Supported categories may include: Hungry, Pain, Tired, and more</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
