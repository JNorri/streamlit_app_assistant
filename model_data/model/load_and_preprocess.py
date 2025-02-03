import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
import sys
warnings.filterwarnings("ignore")

# Проверка путей
def validate_paths():
    base_path = Path("../project_data/model_data/dataset")
    required_dirs = ['train', 'test', 'validation']
    
    missing = []
    for dir in required_dirs:
        if not (base_path / dir).exists():
            missing.append(dir)
    
    if missing:
        print(f"❌ Ошибка: Отсутствуют обязательные директории: {', '.join(missing)}")
        print(f"Проверьте путь: {base_path.resolve()}")
        sys.exit(1)
        
    return base_path

def get_filepaths(base_path):
    """Получение путей к изображениям с проверкой"""
    try:
        train_files = list((base_path / "train").glob(r'**/*.jpg'))
        test_files = list((base_path / "test").glob(r'**/*.jpg'))
        val_files = list((base_path / "validation").glob(r'**/*.jpg'))
        
        if not train_files:
            print("❌ Ошибка: Нет тренировочных изображений в директории train/")
            sys.exit(1)
            
        return train_files, test_files, val_files
        
    except Exception as e:
        print(f"❌ Ошибка при получении файлов: {str(e)}")
        sys.exit(1)

def process_path(fp):
    # Простейшая реализация: преобразование пути в строку
    return str(fp)

def proc_img(filepaths):
    """Обработка путей к изображениям с извлечением меток из структуры директорий"""
    if not filepaths:
        return pd.DataFrame(columns=['Filepath', 'Label'])
    
    # Извлекаем метки из структуры папок
    labels = [fp.parent.name for fp in filepaths]
    
    return pd.DataFrame({
        'Filepath': [str(fp) for fp in filepaths],
        'Label': labels
    }).sample(frac=1).reset_index(drop=True)

def load_data(base_path):
    """Загрузка данных с правильным извлечением меток"""
    train_path = base_path / "train"
    val_path = base_path / "validation"  # Исправлено с val -> validation
    test_path = base_path / "test"

    # Получаем файлы с сохранением структуры директорий
    train_files = list(train_path.glob('**/*.jpg'))
    val_files = list(val_path.glob('**/*.jpg'))
    test_files = list(test_path.glob('**/*.jpg'))

    # Создаем DataFrame с метками
    train_df = proc_img(train_files)
    val_df = proc_img(val_files)
    test_df = proc_img(test_files)

    # Добавьте в конец load_data перед return
    print("Пример тренировочных данных:")
    print(train_df.head())
    print("\nМетки в тренировочных данных:", train_df['Label'].unique())

    return train_df, val_df, test_df

def save_class_names(train_df):
    """Сохранение классов с проверкой"""
    if train_df.empty:
        print("❌ Ошибка: Тренировочные данные пусты")
        sys.exit(1)
        
    class_names = sorted(train_df['Label'].unique())
    class_path = Path("model_data/trained_model/class_names.txt")
    class_path.parent.mkdir(exist_ok=True)
    
    with open(class_path, 'w') as f:
        f.write('\n'.join(class_names))
    
    print(f"✅ Сохранено классов: {len(class_names)}")
    return class_names

def visualize_samples(train_df):
    """Визуализация с проверкой данных"""
    if train_df.empty:
        print("⚠️ Нет данных для визуализации")
        return
    
    df_unique = train_df.drop_duplicates(subset=["Label"])
    num_classes = len(df_unique)
    
    if num_classes == 0:
        print("⚠️ Нет уникальных классов для отображения")
        return
    
    num_samples = min(36, num_classes)
    df_sample = df_unique.sample(num_samples, replace=False)
    
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = rows if rows**2 >= num_samples else rows + 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for ax in axes:
        ax.axis('off')
    
    for ax, (_, row) in zip(axes, df_sample.iterrows()):
        ax.imshow(plt.imread(row['Filepath']))
        ax.set_title(row['Label'], fontsize=8)
        ax.axis('on')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Проверка структуры директорий
    base_path = validate_paths()
    
    # Получение файлов
    train_files, test_files, val_files = get_filepaths(base_path)
    
    # Обработка данных
    train_df = proc_img(train_files)
    test_df = proc_img(test_files)
    val_df = proc_img(val_files)
    
    # Сохранение классов
    class_names = save_class_names(train_df)
    
    # Вывод статистики
    print("\n📊 Статистика датасета:")
    print(f"Образцов для тренировки: {len(train_df)}")
    print(f"Образцов для тестирования: {len(test_df)}")
    print(f"Образцов для валидации: {len(val_df)}")
    print(f"Уникальных классов: {len(class_names)}\n")
    
    # Визуализация
    visualize_samples(train_df)


    