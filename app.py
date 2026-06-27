import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "lung_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="AI Lung Disease Detection",
    page_icon="🫁",
    layout="centered"
)

st.title("AI-Based Lung Disease Detection")
st.write("Upload a chest X-ray image and the model will predict one of the lung disease classes.")

st.warning(
    "This is an academic prototype only. It is not a medical diagnosis system "
    "and should not be used for real clinical decisions."
)


# -----------------------------
# Load model and class names
# -----------------------------
@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Please run train_model.py first.")
        return None
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        st.error("class_names.json not found. Please run train_model.py first.")
        return []
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)


model = load_trained_model()
class_names = load_class_names()


# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, caption="Chest X-ray image", use_container_width=True)

    if model is not None and class_names:
        img = image.resize((224, 224))
        img_array = np.array(img).astype("float32")
        img_array = np.expand_dims(img_array, axis=0)

        probabilities = model.predict(img_array)[0]

        predicted_index = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100

        st.subheader("Prediction Result")

        if predicted_class.lower() == "normal":
            st.success(f"Prediction: {predicted_class}")
        else:
            st.error(f"Prediction: {predicted_class}")

        st.write(f"Confidence Score: **{confidence:.2f}%**")

        st.subheader("Class Probability Breakdown")

        results_df = pd.DataFrame({
            "Class": class_names,
            "Probability (%)": [round(float(p) * 100, 2) for p in probabilities]
        })

        results_df = results_df.sort_values(by="Probability (%)", ascending=False)

        st.dataframe(results_df, use_container_width=True)

        st.info(
            "The model chooses the class with the highest probability. "
            "Low confidence means the model is unsure and may need more training or better balanced data."
        )