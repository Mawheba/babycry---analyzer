import os
import subprocess
import sys

# This forces the app to install missing tools if they aren't found
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import tensorflow as tf
    import librosa
except ImportError:
    # This will run during the very first boot-up
    install('tensorflow-cpu==2.15.0')
    install('librosa')
    install('numpy<2.0.0')
    install('scikit-learn')
    import tensorflow as tf
    import librosa

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# --- REST OF YOUR APP CODE STARTS HERE ---
st.title("Baby Cry Analyzer")
# ... (the rest of your original app.py code)
