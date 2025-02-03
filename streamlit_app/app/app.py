import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
import os
import time

@st.cache_resource
def load_model_and_classes():
    model_path = Path("./model_data/trained_model/fine_tuned_model.keras")
    class_path = Path("./model_data/trained_model/class_names.txt")

    if not model_path.exists():
        st.error("⚠️ Файл модели не найден!")
        st.stop()

    if not class_path.exists():
        st.error("⚠️ Файл с названиями классов не найден!")
        st.stop()

    with open(class_path) as f:
        class_names = [line.strip() for line in f.readlines()]

    return tf.keras.models.load_model(model_path), class_names

def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return np.expand_dims(img, axis=0)

# 🔤 Полный словарь перевода классов
CLASS_TRANSLATIONS = {
    "apple": "🍏 Яблоко",
    "banana": "🍌 Банан",
    "beetroot": "🍠 Свекла",
    "bell pepper": "🫑 Болгарский перец",
    "cabbage": "🥬 Капуста",
    "capsicum": "🫑 Перец капсикум",
    "carrot": "🥕 Морковь",
    "cauliflower": "🌿 Цветная капуста",
    "chilli pepper": "🌶️ Острый перец",
    "corn": "🌽 Кукуруза",
    "cucumber": "🥒 Огурец",
    "eggplant": "🍆 Баклажан",
    "garlic": "🧄 Чеснок",
    "ginger": "🫚 Имбирь",
    "grapes": "🍇 Виноград",
    "jalepeno": "🌶️ Халапеньо",
    "kiwi": "🥝 Киви",
    "lemon": "🍋 Лимон",
    "lettuce": "🥬 Листовой салат",
    "mango": "🥭 Манго",
    "onion": "🧅 Лук",
    "orange": "🍊 Апельсин",
    "paprika": "🌶️ Паприка",
    "pear": "🍐 Груша",
    "peas": "🍃 Горох",
    "pineapple": "🍍 Ананас",
    "pomegranate": "🍎 Гранат",
    "potato": "🥔 Картофель",
    "raddish": "🔴 Редис",
    "soy beans": "🫘 Соевые бобы",
    "spinach": "🥬 Шпинат",
    "sweetcorn": "🌽 Сахарная кукуруза",
    "sweetpotato": "🍠 Батат",
    "tomato": "🍅 Томат",
    "turnip": "🪴 Репа",
    "watermelon": "🍉 Арбуз",
}

# 🎨 Оформление интерфейса
st.markdown(
    """
    <style>
        .stMetric {
            text-align: center;
            font-size: 22px;
            font-weight: bold;
        }
        .prediction-box {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🍎🥦 Классификатор фруктов и овощей")
model, CLASS_NAMES = load_model_and_classes()

uploaded_file = st.file_uploader("📂 Загрузите изображение...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        img = cv2.imread(tmp_file.name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        with st.spinner("⏳ Обрабатываем изображение..."):  # 🔹 Анимация загрузки

            # 🔄 Предобработка и предсказание
            processed_img = preprocess_image(img)
            predictions = model.predict(processed_img)
            predicted_class = CLASS_TRANSLATIONS.get(CLASS_NAMES[np.argmax(predictions)], "❓ Неизвестный")
            confidence = np.max(predictions)

        # 🎯 Отображение результатов
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="📸 Загруженное изображение", use_container_width=True)
        with col2:
            st.markdown(
                f"""
                <div class="prediction-box">
                    <h2>✅ Предсказание:</h2>
                    <h1 style="color:#ff4b4b;">{predicted_class}</h1>
                    <h3 style="color: black;">🔍 Уверенность: <b>{confidence:.2%}</b></h3>
                </div>
                """,
                unsafe_allow_html=True
            )

    finally:
        os.unlink(tmp_file.name)
