# 🍏 Классификатор овощей/фруктов

## 📌 Описание проекта
Этот проект представляет собой веб-приложение для классификации изображений фруктов и овощей с использованием **TensorFlow** и **Streamlit**. Оно позволяет загружать изображения и получать предсказания о том, к какому классу относится объект на фото.

## 🚀 Функционал
- Загрузка изображения через веб-интерфейс
- Предсказание класса (фрукт или овощ) с использованием обученной модели TensorFlow
- Отображение результата классификации при помощи интерфейса на основе **Streamlit**

## 🛠️ Установка и запуск
### 1. Клонирование репозитория
```bash
git clone https://github.com/JNorri/streamlit_app_assistant.git 
```

### 2. Установка зависимостей
Рекомендуется создать виртуальное окружение:
```bash
python -m venv venv
venv\Scripts\activate  # Для Windows
```

Установите необходимые пакеты:
```bash
pip install -r requirements.txt
```

### 3. Запуск приложения
```bash
streamlit run app.py
```
Streamlit-app: [Predictor](https://appappassistant-bmqt9pkrjgcuacasquew7v.streamlit.app/#6ceacf14)

## 📷 Использование
1. Запустите приложение.
2. Загрузите изображение фрукта или овоща.
3. Дождитесь предсказания.

## 🧠 Обучение модели
Датасет: [Fruits and Vegetables Image Recognition Dataset](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)

Модель была обучена с использованием TensorFlow и библиотеки Keras. Для дообучения модели используйте `train.py`:
```bash
python train.py
```

