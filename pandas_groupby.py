import pandas as pd

data = {
    "day": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
    "hour": [6, 12, 18, 24, 6, 12, 18, 24, 6, 12, 18, 24],
    "price": [32.1, 55.4, 61.2, 28.3, 35.6, 58.9, 64.1, 30.2, 29.8, 52.3, 59.7, 27.1],
    "demand": [1100, 1500, 1600, 900, 1150, 1520, 1650, 920, 1080, 1480, 1580, 880]
}

df = pd.DataFrame(data)

# average price per day
print(df.groupby("day")["price"].mean())

# multiple aggregations at once
print(df.groupby("day")["price"].agg(["mean", "max", "min"]))

# group by multiple columns
print(df.groupby(["day", "hour"])["price"].mean())

# aggregate multiple columns simultaneously after grouping by day
print(df.groupby("day").agg({"price": ["mean", "min"], "demand": ["mean", "min"]}))


import numpy as np

# introduce missing values
df.loc[2, "price"] = np.nan
df.loc[7, "demand"] = np.nan
print(df)
print(df.isnull().sum())    # count missing values per column

df_dropped = df.dropna()  # drop rows with missing values
df_filled = df.fillna(df.mean(numeric_only=True))  # fill missing values with column means
print(df_dropped)
print(df_filled)