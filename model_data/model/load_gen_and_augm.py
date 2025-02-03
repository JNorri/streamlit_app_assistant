import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
from pathlib import Path


# Добавляем корень проекта в PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Теперь можно использовать абсолютные импорты
from model_data.model import MODEL_ROOT  # Импорт из __init__.py

# Конфигурация
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
AUGMENTATION_PARAMS = {
    'rotation_range': 30,
    'zoom_range': 0.15,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.15,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

def create_generators(train_df, val_df, test_df):
    """Создание генераторов данных для обучения, валидации и тестирования"""
    
    # Генератор для тренировочных данных с аугментацией
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        **AUGMENTATION_PARAMS
    )

    # Генератор для тестовых и валидационных данных (без аугментации)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    # Создание генераторов
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=IMAGE_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    val_generator = test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='Filepath',
        y_col='Label',
        target_size=IMAGE_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=IMAGE_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_generator, val_generator, test_generator

def load_pretrained_model():
    """Загрузка предобученной модели MobileNetV2"""
    model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    model.trainable = False
    return model

def plot_training_history(history):
    """Визуализация истории обучения"""
    plt.figure(figsize=(12, 5))
    
    # График точности
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # График потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Сохранение и отображение
    plot_path = MODEL_ROOT / "train_model/training_history.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def save_pretrained_model(model, save_path="./model_data/trained_model/pretrained_model.keras"):
    """Сохранение предобученной модели"""
    # Создание директории, если она не существует
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # Сохранение модели
    model.save(save_path)
    print(f"✅ Модель сохранена в {save_path}")

if __name__ == "__main__":
    # Исправляем импорты для запуска напрямую
    from model_data.model.load_and_preprocess import proc_img, get_filepaths
    base_path = Path("./model_data/dataset")
    # Получение данных
    train_files, test_files, val_files = get_filepaths(base_path)
    train_df = proc_img(train_files)
    val_df = proc_img(val_files)
    test_df = proc_img(test_files)
    
    # Создание генераторов
    train_gen, val_gen, test_gen = create_generators(train_df, val_df, test_df)
    
    # Загрузка предобученной модели
    pretrained_model = load_pretrained_model()
    save_pretrained_model(pretrained_model)