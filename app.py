import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from PIL import Image

# ------------------------------
# Load model and label encoder
# ------------------------------
@st.cache_resource
def load_cnn_model():
    model = load_model("D:\AI\Crop_disease\plant_disease_cnn_model_final.h5")
    with open("D:\AI\Crop_disease\labels_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_cnn_model()

# ------------------------------
# App Config
# ------------------------------
IMG_SIZE = 100
USE_GRAYSCALE = True  # True if trained on grayscale, False if RGB

st.set_page_config(page_title="🌱 Crop Disease Detection", layout="centered")
st.title("🌱 Crop Disease Detection System")
st.markdown(
    """
Upload a leaf image and the AI model will predict the **crop type and disease**.  
Make sure the leaf is clear and occupies most of the image for best results.
"""
)

# ------------------------------
# Sidebar Info
# ------------------------------
st.sidebar.header("ℹ️ About")
st.sidebar.write("Detect plant diseases using a CNN model trained on crop leaf images.")
st.sidebar.write("Built with: Streamlit, TensorFlow, OpenCV, NumPy, Python")
st.sidebar.write("Developer: Pradeep")

# ------------------------------
# Image Upload
# ------------------------------
uploaded_file = st.file_uploader("📤 Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert image to array for model
    img = np.array(img)
    if USE_GRAYSCALE:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    else:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    img = img / 255.0  # normalize

    # ------------------------------
    # Predict
    # ------------------------------
    pred_prob = model.predict(img)
    pred_class = np.argmax(pred_prob)
    pred_label = le.classes_[pred_class]
    confidence = pred_prob[0][pred_class] * 100

    # ------------------------------
    # Display Results
    # ------------------------------
    st.success(f"✅ Predicted: **{pred_label}**")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Confidence scores for all classes:")
    for i, cls in enumerate(le.classes_):
        st.write(f"- {cls}: {pred_prob[0][i]*100:.2f}%")
