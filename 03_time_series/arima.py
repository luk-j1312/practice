import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

np.random.seed(42)

# simulating a week of hourly electricity prices with realistic daily pattern and autocorrelation
n=168
hours = np.arange(n)
daily_pattern = np.tile([-15, -18, -20, -21, -20, -12, -5, 5, 12, 15, 14, 16, 18, 17, 15, 16, 18, 22, 20, 15, 10, 5, -2, -8], 7)
noise = np.random.normal(0,3,n)
prices = 50 + daily_pattern + noise
dates = pd.date_range(start="2024-01-01", periods=n, freq="h")
df = pd.DataFrame({"price": prices}, index=dates)

plt.figure(figsize=(14,4))
plt.plot(df.index, df["price"], color="steelblue", linewidth=1)
plt.title("Simulated Hourly Electricity Prices (over the week)")
plt.ylabel("Price ($/MWh)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig("raw_prices.png")
# plt.show()


# step 1: test for stationarity - augmented dickey-fuller test
# null hypothesis: series has a unit root (non-stationary)
# we want to reject the null - i.e. p-value below 0.05
adf_result = adfuller(df["price"])
# print(f"ADF Statistic: {adf_result[0]:.4f}")
# print(f"p-value: {adf_result[1]:.4f}")
if adf_result[1] < 0.05:
    print("series is stationary --- no differencing needed (d=0)")
else:
    print("series is non-stationary --- differencing needed")
# step 2: ACF and PACF plots reveal what p and q to use
fig, axes = plt.subplots(1,2, figsize=(14,4))
plot_acf(df["price"], lags=48, ax=axes[0])
plot_pacf(df["price"], lags=48, ax=axes[1])
axes[0].set_title("Autocorrelation Function (ACF)")
axes[1].set_title("Partial Autocorrelation Function (PACF)")
plt.tight_layout()
# plt.savefig("acf_pacf.png")
# plt.show()

# step 3: fit ARIMA model based on PACF cutting off around lag 5
# d=0 because series is stationary
# starting simple with p=3, d=0, q=1
model = ARIMA(df["price"], order=(3,0,1))
fitted_model = model.fit()
# print(fitted_model.summary())

# step 4: plot fitted values vs. actual
plt.figure(figsize=(14,4))
plt.plot(df.index, df["price"], color="steelblue", linewidth=1, label="actual")
plt.plot(df.index, fitted_model.fittedvalues, color="red", linewidth=1, label="fitted", alpha=0.7)
plt.title("ARIMA(3,0,1) --- actual vs. fitted")
plt.ylabel("Price ($/MWh)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.savefig("arima_fitted.png")
# plt.show()

# step 5: forecast next 24 hours
forecast = fitted_model.get_forecast(steps=24)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()
# explicitly build forecast datetime index
last_timestamp = df.index[-1]
forecast_index = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=24, freq="h")
forecast_mean.index = forecast_index
forecast_ci.index = forecast_index
plt.figure(figsize=(14,4))
# plot previous 48 hours for context
plt.plot(df.index[-48:], df["price"].iloc[-48:], color="steelblue", linewidth=1, label="actual")
plt.plot(forecast_mean.index, forecast_mean, color="red", linewidth=2, label="forecast")
plt.fill_between(forecast_index, forecast_ci.iloc[:,0], forecast_ci.iloc[:,1], color="red", alpha=0.2, label="95% CI")
plt.title("ARIMA(3,0,1) --- 24 hour forecast")
plt.ylabel("Price ($/MWh)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig("arima_forecast.png")
# plt.show()

from statsmodels.tsa.statespace.sarimax import SARIMAX
# SARIMA(p,d,q)(P,D,Q,s)
# s=24 for hourly data --- 24 hour seasonal cycle
# (1,0,1) for non-seasonal component
# (1,0,1,24) for seasonal component
sarima_model = SARIMAX(df["price"], order=(1,0,1), seasonal_order=(1,0,1,24))
fitted_sarima = sarima_model.fit(disp=False)
print(fitted_sarima.summary())

# forecase next 24 hours with SARIMA
sarima_forecast = fitted_sarima.get_forecast(steps=24)
sarima_mean = sarima_forecast.predicted_mean
sarima_ci = sarima_forecast.conf_int()

# align forecast index
sarima_mean.index = forecast_index
sarima_ci.index = forecast_index

# compare ARIMA vs. SARIMA forecasts side-by-side
fig, axes = plt.subplots(2,1, figsize=(14,8))
# ARIMA forecast
axes[0].plot(df.index[-48:], df["price"].iloc[-48:], color="steelblue", linewidth=1, label="actual")
axes[0].plot(forecast_mean.index, forecast_mean, color="red", linewidth=2, label="ARIMA forecast")
axes[0].fill_between(forecast_index, forecast_ci.iloc[:,0], forecast_ci.iloc[:,1], color="red", alpha=0.2, label="95% CI")
axes[0].set_title("ARIMA(3,0,1) --- 24 hour forecast")
axes[0].set_ylabel("Price ($/MWh)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
# SARIMA forecast
axes[1].plot(df.index[-48:], df["price"].iloc[-48:], color="steelblue", linewidth=1, label="actual")
axes[1].plot(sarima_mean.index, sarima_mean, color="green", linewidth=2, label="SARIMA forecast")
axes[1].fill_between(forecast_index, sarima_ci.iloc[:,0], sarima_ci.iloc[:,1], color="green", alpha=0.2, label="95% CI")
axes[1].set_title("SARIMA(1,0,1)(1,0,1,24) --- 24 hour forecast")
axes[1].set_ylabel("Price ($/MWh)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sarima_comparison.png")
plt.show()