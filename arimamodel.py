
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_excel('data1116.xlsx', engine='openpyxl')

for column in df.columns:
    df[column] = df[column].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
df = df.astype(float)


df['date'] = pd.to_datetime(df.iloc[:, 0].astype(int).astype(str) + df.iloc[:, 1].astype(int).astype(str).str.zfill(3) + df.iloc[:, 2].astype(int).astype(str).str.zfill(2), format='%Y%j%H')


series = df.iloc[:, 24]


plt.figure(figsize=(15, 6))
plt.plot(df['date'], series, label='DST индекс', color='blue')
plt.title('График DST индекса')
plt.xlabel('Время')
plt.ylabel('DST индекс')
plt.legend()
plt.show()

# STL-декомпозиция 

stl = STL(series, seasonal=27, period=27)  
result = stl.fit()
trend, seasonal, residual = result.trend, result.seasonal, result.resid

plt.figure(figsize=(12, 8))
plt.subplot(311)
plt.plot(df['date'], series, label='Исходный ряд')
plt.legend()
plt.subplot(312)
plt.plot(df['date'], trend, label='Тренд', color='green')
plt.legend()
plt.subplot(313)
plt.plot(df['date'], seasonal, label='Сезонная составляющая', color='red')
plt.legend()
plt.show()
series_new = series - trend - seasonal

# Визуализация результата (остаться только остатки)

plt.figure(figsize=(15, 6))
plt.plot(df['date'], series_new, label='Остатки (без тренда и сезонности)', color='purple')
plt.title('Ряд без тренда и сезонности')
plt.xlabel('Время')
plt.ylabel('Значения')
plt.legend()
plt.show()
# строю ACF и PACF пока не дифференцирую

plt.figure(figsize=(15, 6))
plt.suptitle('ACF и PACF до дифференцирования')
plt.subplot(211)
plot_acf(series_new, lags=50, ax=plt.gca())
plt.subplot(212)
plot_pacf(series_new, lags=50, ax=plt.gca())
plt.show()
# дифференцирование
series_diff = series_new.diff().dropna()
adf_result = adfuller(series_diff)
print(f'ADF Statistic: {adf_result[0]:.3f}, p-value: {adf_result[1]:.3f}')
print("Ряд стационарен." if adf_result[1] < 0.05 else "Ряд нестационарен.")

#  ACF и PACF
plt.figure(figsize=(15, 6))
plt.suptitle('ACF и PACF после дифференцирования')
plt.subplot(211)
plot_acf(series_diff, lags=50, ax=plt.gca())
plt.subplot(212)
plot_pacf(series_diff, lags=50, ax=plt.gca())
plt.show()
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


split_index = int(len(series_new) * 0.9)  
train, test = series_new[:split_index], series_new[split_index:]


model = ARIMA(train, order=(1, 1, 1))  # p=1, d=1, q=1 посмотрела по графику ACF и PACF,перед построением модели дифференцировала 1 раз //автоарима не работает на этих данных 
model_fit = model.fit()
print(model_fit.summary())

#прогноз
forecast_residual = model_fit.forecast(steps=len(test))


# возвращаю обратно сезонность и тренд

forecast_trend = trend[-len(test):]
forecast_seasonal = seasonal[-len(test):]
forecast_final = forecast_residual + forecast_trend + forecast_seasonal


# визуализация прогнозов
plt.figure(figsize=(10, 6))
plt.plot(df['date'][:split_index], train + trend[:split_index] + seasonal[:split_index], label='Учебная выборка (train)', color='blue')
plt.plot(df['date'][split_index:], test + trend[split_index:] + seasonal[split_index:], label='Тестовая выборка (test)', color='orange')
plt.plot(df['date'][split_index:], forecast_final, label='Прогноз ARIMA(1, 1, 1)', color='red')
plt.title('Прогноз ARIMA(1, 1, 1) на тестовой выборке с восстановлением тренда и сезонности')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.legend()
plt.show()

# оценка модели
mse = mean_squared_error(test + trend[split_index:] + seasonal[split_index:], forecast_final)
mae = mean_absolute_error(test + trend[split_index:] + seasonal[split_index:], forecast_final)
print(f'Среднеквадратичная ошибка (MSE): {mse:.3f}')
print(f'Средняя абсолютная ошибка (MAE): {mae:.3f}')
