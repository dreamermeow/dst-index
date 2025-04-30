import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
import warnings

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_excel('data1116.xlsx', engine='openpyxl')

def to_float(x):
    return float(x.replace(',', '.')) if isinstance(x, str) else x
for col in df.columns:
    df[col] = df[col].apply(to_float)
df = df.astype(float)

N_FUTURE = 96


full_target_series = df.iloc[:, 24]
stl_full = STL(full_target_series, seasonal=27, period=27)
res_full = stl_full.fit()
trend_full = res_full.trend
seasonal_full = res_full.seasonal


df_future = df.iloc[-N_FUTURE:].copy()
target_future_real = full_target_series.iloc[-N_FUTURE:].copy()

df = df.iloc[:-N_FUTURE].copy()
trend = trend_full[:-N_FUTURE]
seasonal = seasonal_full[:-N_FUTURE]
target_series = full_target_series[:-N_FUTURE]


features_df = df.iloc[:, 3:32].copy()
nan_mask = features_df.isna().any()
nan_idx = np.where(nan_mask)[0]
if len(nan_idx):
    col_positions = (nan_idx + 3).tolist()
    print(f"Найдены пропуски в колонках на позициях: {col_positions}. Удаляю их.")
    features_df.drop(features_df.columns[nan_idx], axis=1, inplace=True)

resid = target_series - trend - seasonal
adf_pvalue = adfuller(resid.dropna())[1]
print(f"ADF p-value: {adf_pvalue:.5f} -> {'stationary' if adf_pvalue < 0.05 else 'non-stationary'}")


scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(features_df.values)
resid_vals = resid.values.reshape(-1, 1)
y_scaled = scaler_y.fit_transform(resid_vals)

# Создание последовательностей
SEQ_LEN = 24
def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)

# train/test
train_size = int(len(X_seq) * 0.80)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# LSTM модель
n_features = X_train.shape[2]
model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(SEQ_LEN, n_features)),
    BatchNormalization(),
    Dropout(0.4),
    LSTM(128, return_sequences=False),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

#Обучение
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

model.save('lstm_dst_model.h5')
print("Модель успешно сохранена в файл lstm_dst_model.h5")
#Прогноз
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

y_train_inv = scaler_y.inverse_transform(y_train).flatten()
train_pred_inv = scaler_y.inverse_transform(train_pred).flatten()
y_test_inv = scaler_y.inverse_transform(y_test).flatten()
test_pred_inv = scaler_y.inverse_transform(test_pred).flatten()

# Восстановление DST-прогноза 
train_full_inv = train_pred_inv + trend[SEQ_LEN:SEQ_LEN + len(train_pred_inv)] + seasonal[SEQ_LEN:SEQ_LEN + len(train_pred_inv)]
test_full_inv  = test_pred_inv + trend[SEQ_LEN + len(train_pred_inv):SEQ_LEN + len(train_pred_inv) + len(test_pred_inv)] + seasonal[SEQ_LEN + len(train_pred_inv):SEQ_LEN + len(train_pred_inv) + len(test_pred_inv)]

# Визуализация результатов 
if 'date' in df.columns:
    dates = df['date']
else:
    dates = pd.RangeIndex(len(df))
dates_train = dates[SEQ_LEN: train_size + SEQ_LEN]
dates_test  = dates[train_size + SEQ_LEN:]
plt.figure(figsize=(14,6))
plt.plot(dates_train, target_series[SEQ_LEN: SEQ_LEN + len(train_full_inv)],  label='Факт Train', alpha=0.6)
plt.plot(dates_train, train_full_inv, label='Прогноз Train', alpha=0.8)
plt.plot(dates_test,  target_series[SEQ_LEN + len(train_full_inv): SEQ_LEN + len(train_full_inv) + len(test_full_inv)], label='Факт Test', alpha=0.6)
plt.plot(dates_test,  test_full_inv,  label='Прогноз Test', alpha=0.8)
plt.legend()
plt.title('Прогноз LSTM: реальные и предсказанные DST')
plt.xlabel('Время')
plt.ylabel('DST индекс')
plt.grid(True)
plt.tight_layout()
plt.show()

#Оценка
mse = mean_squared_error(target_series[train_size + SEQ_LEN:], test_full_inv)
mae = mean_absolute_error(target_series[train_size + SEQ_LEN:], test_full_inv)
correlation = np.corrcoef(target_series[train_size + SEQ_LEN:], test_full_inv)[0, 1]
print(f"MSE: {mse:.3f}, MAE: {mae:.3f}")
print(f"Корреляция: {correlation:.3f}")

#Прогноз 
def forecast_future_autoregressive(model, X_last_seq, n_steps, scaler_y, trend_full, seasonal_full, start_idx):
    future_preds = []
    current_seq = X_last_seq.copy()

    for step in range(n_steps):

        pred_scaled = model.predict(current_seq[np.newaxis, ...], verbose=0)[0][0]
        future_preds.append(pred_scaled)

        next_input = current_seq[-1].copy()

        current_seq = np.vstack([current_seq[1:], next_input])
    future_preds_inv = scaler_y.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    future_with_components = future_preds_inv + trend_full[start_idx:start_idx + n_steps] + seasonal_full[start_idx:start_idx + n_steps]
    return future_with_components

X_last_seq = X_scaled[-SEQ_LEN:]
future_forecast = forecast_future_autoregressive(
    model=model,
    X_last_seq=X_scaled[-SEQ_LEN:], 
    n_steps=N_FUTURE,
    scaler_y=scaler_y,
    trend_full=trend_full,
    seasonal_full=seasonal_full,
    start_idx=len(trend)
)
#Сравнение прогноза с реальностью
mse_future = mean_squared_error(target_future_real, future_forecast)
mae_future = mean_absolute_error(target_future_real, future_forecast)
corr_future = np.corrcoef(target_future_real, future_forecast)[0, 1]
print(f"\n[Оценка на отложенных 96 значениях]")
print(f"MSE: {mse_future:.3f}, MAE: {mae_future:.3f}, Корреляция: {corr_future:.3f}")

# Визуализация сравнения
plt.figure(figsize=(14,6))
plt.plot(target_future_real.index, target_future_real, label='Реальные значения (истина)')
plt.plot(target_future_real.index, future_forecast, label='Прогноз модели', linestyle='--')
plt.title('Сравнение прогноза и реальности (последние 96 значений)')
plt.xlabel('Дата' if 'date' in df.columns else 'Индекс')
plt.ylabel('DST индекс')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

