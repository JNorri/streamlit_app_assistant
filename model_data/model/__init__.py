from pathlib import Path

# Определение корневой директории модели
MODEL_ROOT = Path(__file__).parent.resolve()

# Экспорт основных компонентов
__all__ = [
    'load_and_preprocess',
    'load_gen_and_augm',
    'train_model'
]

# Автоматическая загрузка классов при импорте
try:
    with open(MODEL_ROOT/'train_model'/'class_names.txt') as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    CLASS_NAMES = []

def get_class_names():
    return CLASS_NAMES.copy()