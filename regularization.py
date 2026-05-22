import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# simulating hourly electricity price data with many features
n = 300

# true underlying drivers of price
temperature = np.random.normal(70, 15, n)
demand = np.random.normal(1400, 200, n)
gas_price = np.random.normal(3.5, 0.5, n)
hour_of_day = np.tile(np.arange(24), n // 24 + 1)[:n]
wind_generation = np.random.normal(500, 150, n)

# true price relationship --- only these five variables actually matter
true_price = (0.04*demand + 0.3*temperature + 8.0*gas_price + -0.02*wind_generation 
              + 2.0*np.sin(2*np.pi*hour_of_day/24) 
              + np.random.normal(0, 5, n)  # irreducible noise
              )

# now add 20 noise variables that have NO real relationship with price
# OLS won't know these are noise and will fit them anyway
noise_variables = np.random.normal(0,1, (n,20))
noise_df = pd.DataFrame(noise_variables, columns=[f"noise_{i}" for i in range(20)])

# build full-feature DataFrame
df = pd.DataFrame({"price": true_price, "temperature": temperature, "demand": demand, 
                   "gas_price": gas_price, "hour_of_day": hour_of_day, 
                   "wind_generation": wind_generation})
df = pd.concat([df, noise_df], axis=1)

print(df.shape)
print(df.describe())
