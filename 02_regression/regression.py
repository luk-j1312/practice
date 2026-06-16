import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

np.random.seed(42)

# simulate some hourly data
n = 200
demand = np.random.normal(1400, 200, n)
noise = np.random.normal(0, 8, n)
price = 0.04*demand + noise + 10
df = pd.DataFrame({"demand": demand, "price": price})

# OLS regression --- price as function of demand
df["demand_standardized"] = (df["demand"] - df["demand"].mean()) / df["demand"].std()
X = sm.add_constant(df["demand_standardized"])
Y = df["price"]
model = sm.OLS(Y, X).fit(cov_type="HC3")  # cov_type="HC3" gives robust standard errors
print(model.summary())

# scatter plot with regression line
plt.figure(figsize=(8,5))
plt.scatter(df["demand_standardized"], df["price"], alpha=0.4, color="steelblue", label="observations")

# generating predicted values for the line
x_range = np.linspace(df["demand_standardized"].min(), df["demand_standardized"].max(), 100)
x_range_constant = sm.add_constant(x_range)
y_predicted = model.predict(x_range_constant)
plt.plot(x_range, y_predicted, color="red", linewidth=2, label="OlS regression line")
plt.xlabel("Demand standardized (MWh)")
plt.ylabel("Price ($/MWh)")
plt.title("Electricity Price vs Demand")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("regression.png")
plt.show()

# residual analysis
residuals = model.resid
fitted_values = model.fittedvalues
fig, axes = plt.subplots(1,2, figsize=(12,4))
# residuals vs. fitted values
axes[0].scatter(fitted_values, residuals, alpha=0.4, color="steelblue")
axes[0].set_xlabel("Fitted values")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Residuals vs. Fitted values")
axes[0].grid(True,alpha=0.3)
# histogram of residuals
axes[1].hist(residuals, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
axes[1].set_xlabel("residual")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Distribution of Residuals")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("residuals.png")
plt.show()