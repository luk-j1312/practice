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

# separate features from target
X = df.drop("price", axis=1)
Y = df["price"]

# train/test split --- 80% training, 20% testing
# this is how we simulate out-of-sample prediction performance
split = int(0.8*n)
X_train = X.iloc[:split]
X_test = X.iloc[split:]
Y_train = Y.iloc[:split]
Y_test = Y.iloc[split:]
print(f"training size: {X_train.shape}")
print(f"test size: {X_test.shape}")


# standardize features --- critical for Ridge and Lasso
# because the penalty treats all coefficients equally
# a variable measured in thousands (demand) would be
# penalized much more than one measured in single digits (gas_price)
# unless we put everything on the same scale first
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# fit OLS
ols = LinearRegression()
ols.fit(X_train_scaled, Y_train)
ols_train_pred = ols.predict(X_train_scaled)
ols_test_pred = ols.predict(X_test_scaled)

# fit Ridge --- lambda = 1.0 to start
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, Y_train)
ridge_train_pred = ridge.predict(X_train_scaled)
ridge_test_pred = ridge.predict(X_test_scaled)

# fit lasso --- lambda = 0.1 to start
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, Y_train)
lasso_train_pred = lasso.predict(X_train_scaled)
lasso_test_pred = lasso.predict(X_test_scaled)

# compare train vs. test MSE for each model
results = {
    "OLS":   {"train": mean_squared_error(Y_train, ols_train_pred),
              "test":  mean_squared_error(Y_test, ols_test_pred)},
    "Ridge": {"train": mean_squared_error(Y_train, ridge_train_pred),
              "test":  mean_squared_error(Y_test, ridge_test_pred)},
    "Lasso": {"train": mean_squared_error(Y_train, lasso_train_pred),
              "test":  mean_squared_error(Y_test, lasso_test_pred)}
}

print(f"{'Model':<10} {'Train MSE':<15} {'Test MSE':<15} {'Overfit Gap':<10}")
print("-"*50)
for model, scores in results.items():
    gap = scores["test"] - scores["train"]
    print(f"{model:<10} {scores['train']:<15.3f} {scores['test']:<15.3f} {gap:.3f}")

# examine which coefficients lasso zeroed out
feature_names = X.columns.tolist()
lasso_coefficients = pd.Series(lasso.coef_, index=feature_names)
ridge_coefficients = pd.Series(ridge.coef_, index=feature_names)

print("LASSO coefficients:")
print(lasso_coefficients.round(3))
print(f"\nNumber of features zeroed out by Lasso: {(lasso_coefficients == 0).sum()}")
print(f"Number of features zeroed out by Ridge: {(ridge_coefficients == 0).sum()}")

# separate real vs. noise coefficients
real_features = ["temperature", "demand", "gas_price", "hour_of_day", "wind_generation"]
noise_features = [c for c in feature_names if c.startswith("noise_")]

print("\nLasso coefficients on REAL features:")
print(lasso_coefficients[real_features].round(3))
print("\nLasso coefficients on NOISE features:")
print(lasso_coefficients[noise_features].round(3))

from sklearn.linear_model import RidgeCV, LassoCV

# RidgeCV and LassoCV automatically perform cross-validation
# to find the best alpha from a grid of candidates
# cv=5 means 5-fold cross-validation

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_scaled, Y_train)

lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, Y_train)

print(f"best Ridge alpha: {ridge_cv.alpha_}")
print(f"best Lasso alpha: {lasso_cv.alpha_}")

# evaluate with optimal alphas
ridge_cv_test = mean_squared_error(Y_test, ridge_cv.predict(X_test_scaled))
lasso_cv_test = mean_squared_error(Y_test, lasso_cv.predict(X_test_scaled))

print(f"\nOLS test MSE: {results['OLS']['test']:.3f}")
print(f"Ridge CV test MSE: {ridge_cv_test:.3f}")
print(f"Lasso CV test MSE: {lasso_cv_test:.3f}")

# how many variables did tuned Lasso zero out?
lasso_cv_coefficients = pd.Series(lasso_cv.coef_, index=feature_names)
print(f"\nVariables zeroed by tuned Lasso: {(lasso_cv_coefficients == 0).sum()}")
print("\nTuned Lasso noise coefficients:")
print(lasso_cv_coefficients[noise_features].round(3))

from sklearn.linear_model import lasso_path
alphas_path, coefs_path, _ = lasso_path(X_train_scaled, Y_train, alphas=np.logspace(-3,2,100))
plt.figure(figsize=(12,6))

# plot noise variables in grey, real variables in color
for i, name in enumerate(feature_names):
    if name in real_features:
        plt.plot(-np.log10(alphas_path), coefs_path[i], linewidth=2, label=name)
    else: plt.plot(-np.log10(alphas_path), coefs_path[i], color="lightgrey", linewidth=0.8, alpha=0.7)
plt.axvline(-np.log10(lasso_cv.alpha_), color="black",
            linestyle="--", label=f"chosen alpha={lasso_cv.alpha_}")
plt.xlabel("increasing regularization strength →\n(-log10 alpha)")
plt.ylabel("coefficient value")
plt.title("Lasso Regularization Path\ncolored = real features, grey = noise")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lasso_path.png")
plt.show()