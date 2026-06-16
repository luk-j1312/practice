import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, roc_auc_score)

np.random.seed(42)
n=1000

# features
hour = np.random.randint(0, 24, n)  # hour of day
temperature = np.random.normal(75, 15, n)  # temp in degrees F
load = np.random.normal(50, 10, n)  # grid load in GW
wind = np.random.normal(30, 10, n)  # wind generation in GW

# true log-odds: spikes more likely during high load, low wind, peak hours
log_odds = (-4 + 0.05*load - 0.08*wind + 0.03*temperature 
            + 0.15*((hour>=15) & (hour<=20)).astype(float))  # peak hours
prob_spike = 1/(1+np.exp(-log_odds))
spike = (np.random.uniform(0,1,n) < prob_spike).astype(int)

# DataFrame
df = pd.DataFrame({'hour': hour, 'temperature': temperature, 'load': load, 'wind': wind, 
                   'peak_hour': ((hour>=15) & (hour<=20)).astype(int), 
                   'spike': spike})

print(f"Spike rate: {spike.mean():.1%}")  # should be ~10-20%

# features are target
X = df[['temperature', 'load', 'wind', 'peak_hour']]
Y = df['spike']

# train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# scale (same rule as with regularization: we want all features on the same scale for the model to treat them equally; fit on train only)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# fit logistic regression
model = LogisticRegression(random_state=42)
model.fit(X_train_sc, Y_train)

# coefficients
coef_series = pd.Series(model.coef_[0], index=X.columns)
print("\nCoefficients:")
print(coef_series.sort_values())

# predictions --- two kinds
Y_pred = model.predict(X_test_sc)  # class labels of 0 or 1
Y_prob = model.predict_proba(X_test_sc)[:,1]  # probability of spike (class 1)

# confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

# ROC curve
fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)
auc = roc_auc_score(Y_test, Y_prob)
fig, axes = plt.subplots(1,2,figsize=(12,5))
axes[0].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.3f}')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve — Price Spike Classifier')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Predicted probability distribution by true class
axes[1].hist(Y_prob[Y_test == 0], bins=30, alpha=0.6,
             color='steelblue', label='No spike')
axes[1].hist(Y_prob[Y_test == 1], bins=30, alpha=0.6,
             color='tomato', label='Spike')
axes[1].axvline(0.5, color='black', linestyle='--', lw=1, label='Default threshold')
axes[1].set_xlabel('Predicted Probability of Spike')
axes[1].set_ylabel('Count')
axes[1].set_title('Predicted Probabilities by True Class')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("classification.png", dpi=150, bbox_inches="tight")
plt.show()