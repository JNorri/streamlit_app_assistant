import tensorflow as tf
from pathlib import Path

# Пути к файлам
PRETRAINED_MODEL_PATH = Path("./model_data/trained_model/pretrained_model.keras")
FINE_TUNED_MODEL_PATH = Path("./model_data/trained_model/fine_tuned_model.keras")

# Загрузка предобученной модели
if PRETRAINED_MODEL_PATH.exists():
    pretrained_model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)
    print(f"✅ Загружена предобученная модель из {PRETRAINED_MODEL_PATH.resolve()}")
else:
    raise FileNotFoundError(f"❌ Файл {PRETRAINED_MODEL_PATH} не найден!")

# Добавление слоев для дообучения
inputs = pretrained_model.input
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(36, activation='softmax')(x)

# Создание полной модели
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Компиляция модели
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Загружаем генераторы данных
from load_gen_and_augm import create_generators
from load_and_preprocess import load_data

base_path = Path("./model_data/dataset")
train_df, val_df, test_df = load_data(base_path)
train_gen, val_gen, test_gen = create_generators(train_df, val_df, test_df)

# Обучение модели
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )
    ]
)

# Сохранение дообученной модели
model.save(FINE_TUNED_MODEL_PATH)
print(f"✅ Дообученная модель сохранена в {FINE_TUNED_MODEL_PATH.resolve()}")
