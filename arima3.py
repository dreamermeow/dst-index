#наиболее похожий на правду вариант 
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_excel('data1116.xlsx', engine='openpyxl')

for column in df.columns:
    df[column] = df[column].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
df = df.astype(float)


series = df.iloc[:, 24]

# Построение графика временного ряда
plt.figure(figsize=(15, 8))
plt.plot(series, label='DST индекс', color='blue')
plt.title('График DST индекса')
plt.xlabel('Наблюдения')
plt.ylabel('DST индекс')
plt.legend()
plt.show()

# ================================
# 2. Предобработка и диагностика
# ================================

# KPSS-тест для проверки стационарности
def kpss_test(series):
    kpss_stat, p_value, _, _ = kpss(series, regression='c', nlags="auto")
    print(f'KPSS Test Statistic: {kpss_stat:.3f}, p-value: {p_value:.3f}')
    if p_value < 0.05:
        print("Ряд не является стационарным (по KPSS-тесту).")
    else:
        print("Ряд стационарен (по KPSS-тесту).")

kpss_test(series)

# (Опционально) Удаление выбросов: оставляем значения между 1%- и 99%-квантилями
q_low = series.quantile(0.01)
q_high = series.quantile(0.99)
series_clean = series[(series > q_low) & (series < q_high)]

# Построение графиков ACF и PACF
plt.figure(figsize=(12,6))
plt.subplot(211)
plot_acf(series, lags=50, ax=plt.gca())
plt.subplot(212)
plot_pacf(series, lags=50, ax=plt.gca())
plt.show()

# Проверка стационарности после первого дифференцирования
diff_series = series.diff().dropna()
adf_result = adfuller(diff_series)
print(f'ADF Statistic (после дифференцирования): {adf_result[0]:.3f}, p-value: {adf_result[1]:.3f}')
if adf_result[1] < 0.05:
    print("Ряд стационарен после первого дифференцирования.")
else:
    print("Ряд все еще нестационарен, возможно, потребуется второе дифференцирование.")



model_auto = auto_arima(series, seasonal=False, trace=True, stepwise=True)
print(model_auto.summary())

# ================================
# 4. Разделение данных на обучающую и тестовую выборки
# ================================

# Используем последние 10 наблюдений для тестирования
forecast_steps = 10
train = series[:-forecast_steps]
test = series[-forecast_steps:]
print(f"Размер обучающей выборки: {len(train)}, Размер тестовой выборки: {len(test)}")



# Создаем и обучаем модель SARIMAX (ARIMA(0,1,1) по результатам auto_arima)
model = SARIMAX(train, order=(0, 1, 1))
results = model.fit(disp=False)
print(results.summary())

# Прогнозирование для тестовой выборки
forecast_obj = results.get_forecast(steps=forecast_steps)
forecast_mean = forecast_obj.predicted_mean
conf_int = forecast_obj.conf_int()

# Если индекс не является DatetimeIndex, создаем числовой индекс для прогнозных значений:
if not isinstance(train.index, pd.DatetimeIndex):
    forecast_index = range(len(train), len(train) + forecast_steps)
    forecast_mean.index = forecast_index
    conf_int.index = forecast_index
else:
    forecast_index = forecast_mean.index



mse = mean_squared_error(test, forecast_mean)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test, forecast_mean)
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")


plt.figure(figsize=(12, 6))
# Обучающие данные
plt.plot(train.index, train, label='Обучающие данные', color='blue')
# Фактические тестовые данные
plt.plot(test.index, test, label='Фактические тестовые данные', color='green')
# Прогноз
plt.plot(forecast_index, forecast_mean, label='Прогноз', color='red', marker='o')
# Доверительный интервал
plt.fill_between(forecast_index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='pink', alpha=0.3)

plt.xlabel('Наблюдения')
plt.ylabel('DST индекс')
plt.title('Прогноз ARIMA(0,1,1) и сравнение с тестовой выборкой')
plt.legend()
plt.show()
