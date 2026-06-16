import matplotlib.pyplot as plt
import pandas as pd

data = {
    "hour": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    "price": [28.3, 25.1, 23.4, 22.8, 24.2, 31.5, 42.3, 55.6,
              58.2, 56.4, 54.8, 57.3, 59.1, 58.7, 57.2, 58.9,
              62.4, 68.3, 65.1, 58.4, 48.2, 42.1, 36.8, 31.2]
}

df = pd.DataFrame(data)

# line graph
plt.figure(figsize=(12, 5))
plt.plot(df["hour"], df["price"], color="steelblue", linewidth=2)
plt.title("Hourly Electricity Prices")
plt.xlabel("Hour of day")
plt.ylabel("Price ($/MWh)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig("hourly_electricity_prices.png")
# plt.show()

# bar chart
df["period"] = pd.cut(df["hour"], bins=[0, 6, 12, 18, 24], labels=["night", "morning", "afternoon", "evening"])
period_avg = df.groupby("period", observed=True)["price"].mean()
plt.figure(figsize=(8,5))
period_avg.plot(kind="bar", color="steelblue", edgecolor="white")
plt.title("Average Price by Time of Day")
plt.xlabel("Period")
plt.ylabel("Average price ($/MWh)")
plt.xticks(rotation=0)
plt.tight_layout()
# plt.savefig("period_prices.png")
# plt.show()

# histogram
plt.figure(figsize=(8,5))
plt.hist(df["price"], bins=10, color="steelblue", edgecolor="white", alpha=0.8)
plt.title("Distribution of Hourly Prices")
plt.xlabel("Price ($/MWh)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("price_distribution.png")
plt.show()