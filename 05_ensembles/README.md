# Energy Analytics & Machine Learning

Self-led study repository covering Python, ML, modeling, and optimization, built in preparation for future coursework and potential work in energy.

**Author:** Luke
**Program:** Princeton BSE, ORFE
**Stack:** Python 3.11 / NumPy / Pandas / Matplotlib / Scikit-learn / Statsmodels / SciPy / PuLP / Pyomo

---

## Contents

### 01 - Python Basics
Core Python syntax, data structures, DataFrames, and visualization. Covers variables, loops, functions, filtering, groupby aggregation, and plotting patterns implemented henceforth in the repository.

### 02 - Regression
OLS regression with STatsmodels: model-fitting, coefficient interpretation, residual diagnostics, and the Durbin-Watson stat. Ridge and Lasso regularization via Scikit-learn: bias-variance tradeoff, cross-validation, regularization paths, and the geometric intuition behind squared vs. absolute value penalties.

### 03 - Time Series
Stationarity testing (Augmented Dickey-Fuller), ACF/PACF diagnostics, ARIMA and SARIMA modeling on synthetic hourly electricity price data. Fourier feature encoding for cyclical variables (sin/cos hour-of-day).

### 04 - Classification
Logistic regression, Linear Discriminant Analysis (LDA), and Quadratic Discriminant Analysis (QDA) applied to electricity price spike prediction. Confusion matrix, precision/recall tradeoff, ROC/AUC evaluation, and the class imbalance problem.

### 05 - Ensemble Methods
Decision tress (Gini impurity, depth tuning, bias-variance visualization). Random forests (bootstrap sampling, random feature subsampling). Gradient boosting (sequential residual fitting, learning rate and n_estimators interaction). All models evaluated via walk-forward cross-validation.

### 06 - Unsupervised Learning
Principal Component Analysis: eigendecompisition of the covariance matrix, explained variance, loadings interpretation. K-means clustering: WCSS minimization, elbow method, silhouette scores, operating regime identification in energy data.

### 07 - Optimization
Unconstrained und constrained optimization with SciPy (SLSQP, BFGS). Linear programming with PuLP (CBC solver): battery arbitrage dispatch, shadow prices, marginal capacity value. Mixed-integer linear programming with Pyomo (GLPK): unit commitment with binary on/off variables and startup costs.

### 08 - Applications
Real hourly electricity demand data from the EIA API (ERCOT, 2023 if I recall the correct year). Feature engineering for energy time series: lag features, rolling stats, multi-period Fourier encoding (for seasonality). Walk-forward cross-validation. Bayesian inference and MC sim for battery revenue valuation under price uncertainty. Neural network basics with Scikit-learn MLP.

---

## Key Concepts by File

| File | Key concepts |
|---|---|
| `regression.py` | OLS, HC3 standard errors, R², Durbin-Watson |
| `regularization.py` | Ridge, Lasso, ElasticNet, RidgeCV, LassoCV |
| `arima.py` | ARIMA(p,d,q), SARIMA, ACF/PACF, ADF test |
| `log_classification.py` | Logistic regression, AUC, precision/recall |
| `lda_classification.py` | LDA, QDA, Bayesian classification |
| `decision_tree.py` | Gini impurity, depth CV, feature importance |
| `random_forest.py` | Bootstrap, feature subsampling, OOB |
| `gradient_boosting.py` | Residual fitting, staged_predict_proba |
| `pca.py` | Eigendecomposition, scree plot, loadings |
| `k_means_clustering.py` | WCSS, elbow, silhouette, operating regimes |
| `linear_prog.py` | LP standard form, simplex, shadow prices |
| `dispatch.py` | MILP, unit commitment, big-M constraints |
| `eia_data.py` | EIA API, walk-forward CV, demand forecasting |
| `bayesian.py` | Beta-Binomial updating, credible intervals, MC |
| `neural_net.py` | Forward pass, ReLU, backprop, early stopping |