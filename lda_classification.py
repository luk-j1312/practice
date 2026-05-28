import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

np.random.seed(42)
n = 1000

# Same data generating process as before
hour = np.random.randint(0, 24, n)
temp = np.random.normal(75, 15, n)
load = np.random.normal(50, 10, n)
wind = np.random.normal(30, 10, n)

log_odds = (-4
            + 0.05 * load
            - 0.08 * wind
            + 0.03 * temp
            + 0.15 * ((hour >= 15) & (hour <= 20)).astype(float))

prob_spike = 1 / (1 + np.exp(-log_odds))
spike = (np.random.uniform(0, 1, n) < prob_spike).astype(int)

df = pd.DataFrame({
    "temp": temp, "load": load,
    "wind": wind,
    "peak_hour": ((hour >= 15) & (hour <= 20)).astype(int),
    "spike": spike
})

X = df.drop("spike", axis=1)
Y = df["spike"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# fit LDA, QDA, and logit
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_sc, Y_train)
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_sc, Y_train)
logit = LogisticRegression(random_state=42)
logit.fit(X_train_sc, Y_train)

# compare via five-fold CV AUC
models = {"LDA": lda, "QDA": qda, "Logistic Regression": logit}
print("five-fold CV AUC scores:")
print("-"*35)
for name, model in models.items():
    scores = cross_val_score(model, X_train_sc, Y_train, cv=5, scoring="roc_auc")
    print(f"{name:<25} {scores.mean():.3f} (+/- {scores.std():.3f})")

# LDA: inspect class means
print("\nLDA class means (scaled features):")
means_df = pd.DataFrame(lda.means_, columns=X.columns, index=["no spike", "spike"])
print(means_df.round(3))

# plot: predicted probability distributions for LDA vs. Logit
fig, axes = plt.subplots(1,2,figsize=(12,5))
for ax, (name, model) in zip(axes, {"LDA": lda, "Logistic Regression": logit}.items()):
    probs = model.predict_proba(X_test_sc)[:, 1]
    ax.hist(probs[Y_test == 0], bins=30, alpha=0.6,
            color="steelblue", label="no spike")
    ax.hist(probs[Y_test == 1], bins=30, alpha=0.6,
            color="tomato", label="spike")
    ax.axvline(0.5, color="black", linestyle="--", lw=1)
    auc = roc_auc_score(Y_test, probs)
    ax.set_title(f"{name}  |  AUC = {auc:.3f}")
    ax.set_xlabel("Predicted Probability of Spike")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lda_comparison_png", dpi=150, bbox_inches="tight")
plt.show()