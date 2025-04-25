# в этом варианте модели, я рассматриваю только дст-индекс ( без других данных), заранее предобрабатываю как в модели Arima, убирая сезоность и тренд
# то есть тут вместо arima просто lstm слои
# в данной модели MAE = 2.6, когда в Arima было MAE = 7
# по графику также видно точность прогноза, возможны улучшения модели, могу попробовать, а также вариант где я использую другие данные будут в другом файле   


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
import warnings


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_excel('data1116.xlsx', engine='openpyxl')
for col in df.columns:
    df[col] = df[col].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
df = df.astype(float)
df['date'] = pd.to_datetime(
    df.iloc[:, 0].astype(int).astype(str) + 
    df.iloc[:, 1].astype(int).astype(str).str.zfill(3) + 
    df.iloc[:, 2].astype(int).astype(str).str.zfill(2),
    format='%Y%j%H'
)


series = df.iloc[:, 24]

stl = STL(series, seasonal=27, period=27)
res = stl.fit()
trend = res.trend
seasonal = res.seasonal
resid = series - trend - seasonal


adf_pvalue = adfuller(resid.dropna())[1]
print(f"ADF p-value: {adf_pvalue:.5f} -> {'stationary' if adf_pvalue<0.05 else 'non-stationary'}")


scaler = MinMaxScaler(feature_range=(0,1))
resid_vals = resid.values.reshape(-1,1)
resid_scaled = scaler.fit_transform(resid_vals)


def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, 0])
        y.append(data[i+seq_len, 0])
    return np.array(X), np.array(y)

SEQ_LEN = 27

train_size = int(len(resid_scaled) * 0.9)
train_scaled = resid_scaled[:train_size]
test_scaled  = resid_scaled[train_size - SEQ_LEN:]

X_train, y_train = create_sequences(train_scaled, SEQ_LEN)
X_test, y_test   = create_sequences(test_scaled, SEQ_LEN)

X_train = X_train.reshape(-1, SEQ_LEN, 1)
X_test  = X_test.reshape(-1, SEQ_LEN, 1)


def build_model():
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQ_LEN,1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_model()
model.summary()


early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)


history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)


train_pred = model.predict(X_train)
test_pred  = model.predict(X_test)


y_train_inv = scaler.inverse_transform(y_train.reshape(-1,1)).flatten()
train_pred_inv = scaler.inverse_transform(train_pred).flatten()
y_test_inv  = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
test_pred_inv  = scaler.inverse_transform(test_pred).flatten()

def restore(series_part, trend_series, seasonal_series, start_idx):
    end = start_idx + len(series_part)
    return series_part + trend_series.iloc[start_idx:end].values + seasonal_series.iloc[start_idx:end].values

train_restored = restore(train_pred_inv, trend, seasonal, SEQ_LEN)
train_true     = restore(y_train_inv,    trend, seasonal, SEQ_LEN)
test_restored  = restore(test_pred_inv,  trend, seasonal, train_size)
test_true      = restore(y_test_inv,     trend, seasonal, train_size)

dates_train = df['date'][SEQ_LEN:train_size]
dates_test  = df['date'][train_size:len(series)]


plt.figure(figsize=(14,6))
plt.plot(dates_train, train_true, label='Истинные значения ( на обучении)', alpha=0.6)
plt.plot(dates_train, train_restored, label='Прогноз ( на обучении)', alpha=0.8)
plt.plot(dates_test,  test_true,  label='Истинные значения (тест)', alpha=0.6)
plt.plot(dates_test,  test_restored,  label='Прогноз (тест)', alpha=0.8)
plt.legend()
plt.title('LSTM-модель')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.show()

mse = mean_squared_error(test_true, test_restored)
mae = mean_absolute_error(test_true, test_restored)
 
print(f"Test MSE: {mse:.3f}, Test MAE: {mae:.3f}")
