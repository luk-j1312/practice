import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# ----------------------------------------------------------------
# PART 1: Download real hourly electricity demand data
# EIA Series: hourly demand by grid region
# We use ERCOT (Texas) as the region
# ----------------------------------------------------------------
API_KEY  = "mv8BpyvYAJXe8TwHGUkET03O7SIvb0Hz6yynhawQ"
BASE_URL = "https://api.eia.gov/v2"
def get_eia_hourly_demand(api_key, region="TEX", start="2023-01-01", end="2023-12-31"):
    """
    Download hourly electricity demand (MWh) for a grid region.
    Regions: TEX (ERCOT), MISO, PJM, CAL (CAISO), SWPP (SPP)
    """
    url = f"{BASE_URL}/electricity/rto/region-data/data/"
    params = {
        "api_key":          api_key,
        "frequency":        "hourly",
        "data[0]":          "value",
        "facets[respondent][]": region,
        "facets[type][]":   "D",        # D = demand
        "start":            start,
        "end":              end,
        "sort[0][column]":  "period",
        "sort[0][direction]": "asc",
        "length":           5000,
        "offset":           0
    }
    all_data = []
    while True:
        response = requests.get(url, params=params)
        data = response.json()
        records = data.get("response", {}).get("data", [])
        if not records:
            break
        all_data.extend(records)
        if len(records) < 5000:
            break
        params["offset"] += 5000
    df = pd.DataFrame(all_data)
    df["period"] = pd.to_datetime(df["period"])
    df = df.set_index("period").sort_index()
    df["demand"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["demand"]]
# load the data
df = get_eia_hourly_demand(API_KEY)
print("=== Raw Data Overview ===")
print(f"Shape:          {df.shape}")
print(f"Date range:     {df.index.min()} to {df.index.max()}")
print(f"Missing values: {df['demand'].isnull().sum()}")
print(f"\nBasic statistics:")
print(df["demand"].describe().round(1))

# ----------------------------------------------------------------
# PART 2: Clean the data
# ----------------------------------------------------------------
# Fill small gaps (<=3 hours) with linear interpolation
# Longer gaps with forward fill
df["demand"] = (df["demand"]
                .interpolate(method="linear", limit=3)
                .ffill())
# Flag and remove clear outliers
# Values more than 5 standard deviations from the rolling mean
rolling_mean = df["demand"].rolling(window=168, center=True).mean()
rolling_std  = df["demand"].rolling(window=168, center=True).std()
z_score      = (df["demand"] - rolling_mean) / rolling_std
outliers     = (z_score.abs() > 5)
print(f"\nOutliers flagged: {outliers.sum()}")
df.loc[outliers, "demand"] = np.nan
df["demand"] = df["demand"].interpolate(method="linear")
print(f"Missing after cleaning: {df['demand'].isnull().sum()}")

# ----------------------------------------------------------------
# PART 3: Exploratory analysis
# ----------------------------------------------------------------
# Distribution characteristics
print(f"\n=== Distribution ===")
print(f"Mean:     {df['demand'].mean():,.0f} MWh")
print(f"Median:   {df['demand'].median():,.0f} MWh")
print(f"Skewness: {df['demand'].skew():.3f}")
print(f"Max:      {df['demand'].max():,.0f} MWh")
print(f"Min:      {df['demand'].min():,.0f} MWh")
# Average by hour and month
df["hour"]    = df.index.hour
df["month"]   = df.index.month
df["weekday"] = df.index.weekday   # 0=Monday, 6=Sunday
df["is_weekend"] = (df["weekday"] >= 5).astype(int)
hourly_avg  = df.groupby("hour")["demand"].mean()
monthly_avg = df.groupby("month")["demand"].mean()

# ----------------------------------------------------------------
# PART 4: Feature engineering for forecasting
# ----------------------------------------------------------------
def make_features(df):
    """
    Build features for demand forecasting.
    Fourier features capture cyclical patterns without raw integers.
    Lag features give the model access to recent history.
    """
    X = pd.DataFrame(index=df.index)
    # Fourier features for daily cycle
    X["sin_hour"] = np.sin(2 * np.pi * df.index.hour / 24)
    X["cos_hour"] = np.cos(2 * np.pi * df.index.hour / 24)
    # Fourier features for weekly cycle
    hour_of_week = df.index.dayofweek * 24 + df.index.hour
    X["sin_week"] = np.sin(2 * np.pi * hour_of_week / (7 * 24))
    X["cos_week"] = np.cos(2 * np.pi * hour_of_week / (7 * 24))
    # Fourier features for annual cycle
    hour_of_year = df.index.dayofyear * 24 + df.index.hour
    X["sin_year"] = np.sin(2 * np.pi * hour_of_year / (365 * 24))
    X["cos_year"] = np.cos(2 * np.pi * hour_of_year / (365 * 24))
    # Calendar indicators
    X["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    X["month"]      = df.index.month
    # Lag features — recent demand history
    X["demand_lag_1"]   = df["demand"].shift(1)    # 1 hour ago
    X["demand_lag_24"]  = df["demand"].shift(24)   # same hour yesterday
    X["demand_lag_168"] = df["demand"].shift(168)  # same hour last week
    # Rolling statistics
    X["demand_roll_24"] = (df["demand"]
                           .shift(1)
                           .rolling(24).mean())    # 24-hour rolling mean
    return X
X = make_features(df)
y = df["demand"]
# Drop rows with NaN from lag creation
valid = X.notna().all(axis=1) & y.notna()
X, y  = X[valid], y[valid]
print(f"\nFeature matrix shape: {X.shape}")
print(f"Features: {list(X.columns)}")

# ----------------------------------------------------------------
# PART 5: Walk-forward cross-validation
# ----------------------------------------------------------------
# TimeSeriesSplit respects temporal order — never trains on the future
tscv = TimeSeriesSplit(n_splits=5)
model   = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
fold_maes  = []
fold_mapes = []
print("\n=== Walk-Forward Cross-Validation ===")
print(f"{'Fold':<6} {'Train size':>12} {'Test size':>10} "
      f"{'MAE':>10} {'MAPE':>10}")
print("-" * 52)
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mape = (np.abs((y_test - y_pred) / y_test)).mean() * 100
    fold_maes.append(mae)
    fold_mapes.append(mape)
    print(f"{fold+1:<6} {len(train_idx):>12,} {len(test_idx):>10,} "
          f"{mae:>10.1f} {mape:>9.2f}%")
print("-" * 52)
print(f"{'Mean':<6} {'':>12} {'':>10} "
      f"{np.mean(fold_maes):>10.1f} "
      f"{np.mean(fold_mapes):>9.2f}%")

# ----------------------------------------------------------------
# PART 6: Feature importance
# ----------------------------------------------------------------
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print(f"\nFeature importances:")
print(importance.round(3))

# ----------------------------------------------------------------
# PART 7: Plots
# ----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# Raw demand time series — one week sample
week = df["demand"].iloc[:168]
axes[0, 0].plot(week.values, color="steelblue", lw=1.5)
axes[0, 0].set_title("Hourly Demand — First Week")
axes[0, 0].set_xlabel("Hour")
axes[0, 0].set_ylabel("Demand (MWh)")
axes[0, 0].grid(True, alpha=0.3)
# Distribution
axes[0, 1].hist(df["demand"].values, bins=80,
                color="steelblue", edgecolor="white", alpha=0.8)
axes[0, 1].axvline(df["demand"].mean(), color="tomato",
                   linestyle="--", lw=2, label="Mean")
axes[0, 1].axvline(df["demand"].median(), color="seagreen",
                   linestyle="--", lw=2, label="Median")
axes[0, 1].set_title("Demand Distribution")
axes[0, 1].set_xlabel("Demand (MWh)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
# Average by hour (weekday vs weekend)
weekday_hourly = df[df["is_weekend"]==0].groupby("hour")["demand"].mean()
weekend_hourly = df[df["is_weekend"]==1].groupby("hour")["demand"].mean()
axes[1, 0].plot(weekday_hourly, color="steelblue",
                lw=2, label="Weekday")
axes[1, 0].plot(weekend_hourly, color="tomato",
                lw=2, label="Weekend")
axes[1, 0].set_title("Average Demand by Hour")
axes[1, 0].set_xlabel("Hour of Day")
axes[1, 0].set_ylabel("Avg Demand (MWh)")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
# Monthly average
axes[1, 1].bar(range(1, 13), monthly_avg.values,
               color="steelblue", edgecolor="white")
axes[1, 1].set_title("Average Demand by Month")
axes[1, 1].set_xlabel("Month")
axes[1, 1].set_ylabel("Avg Demand (MWh)")
axes[1, 1].set_xticks(range(1, 13))
axes[1, 1].set_xticklabels(["JAN","FEB","MAR","APR","MAY","JUN",
                             "JUL","AUG","SEP","OCT","NOV","DEC"])
axes[1, 1].grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("real_data_eda.png", dpi=150, bbox_inches="tight")
plt.show()