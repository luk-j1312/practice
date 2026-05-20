import pandas as pd

data = {
    "hour": [0, 1, 2, 3, 4, 5],
    "price": [10.99, 5.99, 3.50, 11.2, 7.99, 2.5],
    "demand": [1200, 1100, 1350, 1500, 1280, 1420]
}

df = pd.DataFrame(data)
df.to_csv("prices.csv", index=False)
print(df)
print(df.shape)
print(df.describe())

df = pd.read_csv("prices.csv")
print(df.head())
print(df.tail())
print(df.columns)
print(df.dtypes)

print(df["price"])
print(df["price"].mean())
print(df[["hour", "price"]])

high_price_hours = df[df["price"] > 10]
print(high_price_hours)

busy_expensive = df[(df["price"] > 10) & (df["demand"] > 1400)]
print(busy_expensive)

df["price_per_unit_demand"] = (df["price"] / df["demand"]) * 100
print(df)