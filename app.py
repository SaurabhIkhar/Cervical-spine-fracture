import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
MODEL_PATH = 'model_2_denseNet121.h5'
model = load_model(MODEL_PATH)

# Define class labels (update with your labels)
CLASS_LABELS = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

# Function to preprocess image
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.resize((128, 128))  # Resize as per your model's input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app
st.title("Cervical Spine Fracture Detection")
st.write("Upload a CT scan image to predict fractures.")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("Processing...")
    
    # Preprocess the uploaded image
    img_array = preprocess_image(uploaded_file)
    
    # Predict using the model
    predictions = model.predict(img_array)
    results = {CLASS_LABELS[i]: float(predictions[0][i]) for i in range(len(CLASS_LABELS))}
    
    st.write("### Prediction Results:")
    for class_label, score in results.items():
        st.write(f"{class_label}: {score:.4f}")
