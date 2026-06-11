import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load Model
model = load_model("model.h5")

# Class Labels
class_labels = ["meningioma", "glioma", "notumor", "pituitary"]

st.set_page_config(page_title="Brain Tumor Prediction", layout="centered")

st.title("🧠 Brain Tumor Prediction System")
st.write("Upload an MRI image to predict the tumor type.")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    img = img.resize((200, 200))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array)

    predicted_class_index = np.argmax(pred)

    predicted_class = class_labels[predicted_class_index]
    confidence = np.max(pred) * 100

    st.success(f"🧠 Tumor Type: {predicted_class}")
    st.info(f"🎯 Confidence: {confidence:.2f}%")