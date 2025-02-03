import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
import os


@st.cache_resource
def load_model_and_classes():
    model_path = Path("../../model_data/model/trained_model/.keras")
    class_path = Path("../../model_data/model/trained_model/class_names.txt")
    
    if not model_path.exists():
        st.error("Model file not found!")
        st.stop()
    
    if not class_path.exists():
        st.error("Class labels not found!")
        st.stop()
    
    with open(class_path) as f:
        class_names = [line.strip() for line in f.readlines()]
    
    return tf.keras.models.load_model(model_path), class_names


def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return np.expand_dims(img, axis=0)

st.title("🍎🥦 Классификатор фруктов/овощей")
model, CLASS_NAMES = load_model_and_classes()

uploaded_file = st.file_uploader("Загрузите изображение...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        img = cv2.imread(tmp_file.name)
    
    try:
        # Предобработка и предсказание
        processed_img = preprocess_image(img)
        predictions = model.predict(processed_img)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = np.max(predictions)
        
        # Отображение результатов
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Загруженное изображение", use_column_width=True)
        with col2:
            st.success(f"**Предсказание:** {predicted_class}")
            st.metric("Уверенность", f"{confidence:.2%}")
    
    finally:
        os.unlink(tmp_file.name)