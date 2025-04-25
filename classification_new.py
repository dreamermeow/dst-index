import numpy as np
import pandas as pd
import keras
from keras import layers
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, precision_recall_curve
import tensorflow as tf
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
# Загрузка и подготовка данных
df = pd.read_excel('data1116.xlsx', engine='openpyxl')
for column in df.columns:
    df[column] = df[column].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
df = df.astype(float)

dst_index = df.iloc[:, 24].values
shtorm = np.where(dst_index < -50, 1, 0)
df = df.drop(df.columns[24], axis=1)
features = df.iloc[:, 3:32].values

# Разделение данных
num_val_samples = int(len(features) * 0.2)
train_X, val_X = features[:-num_val_samples], features[-num_val_samples:]
train_y, val_y = shtorm[:-num_val_samples], shtorm[-num_val_samples:]

# Обработка NaN
if np.isnan(train_X).sum() > 0 or np.isnan(val_X).sum() > 0:
    print("Найдены NaN. Замена на средние значения.")
    train_X = np.nan_to_num(train_X, nan=np.nanmean(train_X))
    val_X = np.nan_to_num(val_X, nan=np.nanmean(train_X))

# Балансировка классов
unique, counts = np.unique(train_y, return_counts=True)
class_weights = {
    0: 1,
    1: 5
}

mean, std = np.mean(train_X, axis=0), np.std(train_X, axis=0)
std[std == 0] = 1
train_X = (train_X - mean) / std
val_X = (val_X - mean) / std

model = keras.Sequential([
    layers.Input(shape=train_X.shape[1:]),

    layers.Dense(256, activation="relu", kernel_initializer='he_normal'),
    #layers.Dense(64, activation="relu", kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l1_l2(l1=0.0005, l2=0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(128, activation="relu"),
    #layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(64, activation="relu"),
    #layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),

    layers.Dense(1, activation="sigmoid")
])
'''
model = keras.Sequential([
    layers.Input(shape=train_X.shape[1:]),
    layers.Dense(512, activation="relu", kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l1_l2(l1=0.0005, l2=0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),  
    layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),  
    layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dense(1, activation="sigmoid"),  
])'''
# Компиляция
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.AUC(name='auc'),
        keras.metrics.Recall(),
        keras.metrics.Precision()
    ]
)

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_recall',
    patience=15,
    mode='max',
    restore_best_weights=True
)

# Обучение
history = model.fit(
    train_X, train_y,
    batch_size=128,
    epochs=50,
    validation_data=(val_X, val_y),
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=2
)

# Оптимизация порога
probs = model.predict(val_X).flatten()
precision, recall, thresholds = precision_recall_curve(val_y, probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Оптимальный порог классификации: {optimal_threshold:.2f}")

# Предсказания
predictions = (probs > optimal_threshold ).astype("uint8")

# Матрица ошибок
tn, fp, fn, tp = confusion_matrix(val_y, predictions, labels=[0, 1]).ravel()
print("Правильно предсказанные штормы (True Positives):", tp)
print("Штормы, которые не были предсказаны (False Negatives):", fn)
print("Не штормы, но предсказаны как штормы (False Positives):", fp)
print("Правильно предсказанные не штормы (True Negatives):", tn)

# Метрики
precision = tp / (tp + fp + 1e-7)
recall = tp / (tp + fn + 1e-7)
f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
''' #здесь я смотрю что будет с разными порогами, потому что сверху подбираю самый лучший 
thresholds = np.arange(0.2, 0.5, 0.02)

for threshold in thresholds:
    preds = (probs > threshold).astype("uint8")
    tn, fp, fn, tp = confusion_matrix(val_y, preds, labels=[0, 1]).ravel()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    print(f"Порог: {threshold:.2f} | TP: {tp} | FN: {fn} | FP: {fp} | TN: {tn} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}")
'''