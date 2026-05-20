import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# creating a time series --- seven days of hourly prices
dates = pd.date_range(start="2024-01-01", periods=168, freq="h")

# simulating realistic electricity prices (base + daily pattern + white noise)
np.random.seed(42)
base_price = 45
daily_pattern = np.tile([
    -15, -18, -20, -21, -20, -12,
    -5, 5, 12, 15, 14, 16,
    18, 17, 15, 16, 18, 22,
    20, 15, 10, 5, -2, -8
], 7)
noise = np.random.normal(0,3,168)
prices = base_price + daily_pattern + noise
df = pd.DataFrame({"datetime": dates, "price": prices})

# set datetime as index (standard of course for time series)
df = df.set_index("datetime")
print(df.head(10))
print(df.index)

# resampling to convert from hourly to daily averages
daily_avg = df["price"].resample("D").mean()
print(daily_avg)

# resampling to 6-hour blocks
six_hour_avg = df["price"].resample("6h").mean()
print(six_hour_avg)

# extracting time components
df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek
df["is_weekend"] = df["day_of_week"] >= 5
# get average price across each hour
hourly_avg = df.groupby("hour")["price"].mean()
# average price weekday vs. weekend
print(df.groupby("is_weekend")["price"].mean())

fig, axes = plt.subplots(2, 1, figsize=(14, 8))


# top panel --- full week of prices
axes[0].plot(df.index, df["price"], color="steelblue", linewidth=1)
axes[0].set_title("One Week of Hourly Electricity Prices")
axes[0].set_ylabel("Price ($/MWh)")
axes[0].grid(True, alpha=0.3)

# bottom panel --- average daily prices
axes[1].plot(hourly_avg.index, hourly_avg.values, color="darkorange", linewidth=2)
axes[1].set_title("Average Price by Hour of Day")
axes[1].set_xlabel("Hour")
axes[1].set_ylabel("Average Price ($/MWh)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("timeseries.png")
plt.show()

