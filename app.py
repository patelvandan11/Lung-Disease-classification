import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load trained model
@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("model1.h5")  # Update with actual model path

model = load_model()

# Define class labels
CLASS_NAMES = [ "COVID-19","Normal",  "Pneumonia","Tuberculosis"]

def preprocess_image(image):
    """Preprocess uploaded image to match model input."""
    image = image.convert("RGB")  # Ensure it's in RGB format
    image = image.resize((224, 224))  # Resize to match VGG16 input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image):
    """Make a prediction using the trained model."""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)[0]  # Get first (only) prediction
    return predictions

# Streamlit UI
st.title("LungScanAI: Lung Disease Classification")
st.write("Upload a chest X-ray image to classify it into Normal, Tuberculosis, Pneumonia, or COVID-19.")

uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
    
    # Predict
    with st.spinner("Classifying..."):
        predictions = predict(image)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
    
    # Display result
    st.success(f"*Prediction:* {predicted_class} \n *Confidence:* {confidence:.2f}%")
    
    # Show confidence scores
    st.subheader("Confidence Scores:")
    for class_name, score in zip(CLASS_NAMES, predictions):
        st.write(f"{class_name}: {score * 100:.2f}%")   