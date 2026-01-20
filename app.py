@st.cache_resource
def load_assets():
    try:
        if not os.path.exists('infant_cry_classification_model.h5'):
            raise FileNotFoundError("Model file missing")
        if not os.path.exists('label_encoder.pkl'):
            raise FileNotFoundError("Label encoder missing")
        
        # Fix model loading with custom objects for legacy InputLayer
        custom_objects = {'InputLayer': tf.keras.layers.Input}
        model = load_model('infant_cry_classification_model.h5', 
                          compile=False, 
                          custom_objects=custom_objects)
        with open('label_encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, encoder
    except Exception as e:
        st.error(f"Loading failed: {str(e)}")  # Fixed: no citations
        return None, None
