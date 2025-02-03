import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Пути к данным
DATASET_DIR = os.path.join(BASE_DIR, 'project_data', 'model_data', 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
VALIDATION_DIR = os.path.join(DATASET_DIR, 'validation')

# Гиперпараметры
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Модели
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'model', 'train_model', 'model.h5')
