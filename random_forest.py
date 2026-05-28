import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

np.random.seed(42)
n=1000

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
    'temp': temp, 'load': load,
    'wind': wind,
    'peak_hour': ((hour >= 15) & (hour <= 20)).astype(int),
    'spike': spike
})

X = df.drop("spike", axis=1)
Y = df["spike"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- n_estimators experiment: how many trees do you need? ---
tree_counts = [1, 5, 10, 25, 50, 100, 200, 500]
cv_aucs = []
for n_trees in tree_counts:
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X_train, Y_train, cv=5, scoring='roc_auc')
    cv_aucs.append(scores.mean())
    print(f"n_trees={n_trees:<5}  CV AUC={scores.mean():.3f}")

# --- Fit final model ---
best_rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
best_rf.fit(X_train, Y_train)
# training vs test AUC
train_auc = roc_auc_score(Y_train, best_rf.predict_proba(X_train)[:,1])
test_auc = roc_auc_score(Y_test, best_rf.predict_proba(X_test)[:,1])
print(f"\nRandom Forest (200 trees)")
print(f"Training AUC : {train_auc:.3f}")
print(f"Test AUC     : {test_auc:.3f}")
# Feature importance
importance = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(f"\nFeature importances:\n{importance.round(3)}")

# --- plots ---
fig, axes = plt.subplots(1,2, figsize=(13,5))

# AUC vs. number of trees
axes[0].plot(tree_counts, cv_aucs, 'o-', color='steelblue')
axes[0].axhline(0.68, color='tomato', linestyle='--', lw=1.5, 
                label='Single tree CV AUC (0.68)')
axes[0].axhline(0.747, color='green', linestyle='--', lw=1.5, 
                label='Logistic regression CV AUC (0.747)')
axes[0].set_xscale('log')
axes[0].set_xlabel('Number of Trees (log scale)')
axes[0].set_ylabel('CV AUC')
axes[0].set_title('Random Forest: AUC vs Number of Trees')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
# Feature importance
importance.plot(kind='bar', ax=axes[1],
                color='steelblue', edgecolor='white')
axes[1].set_title('Random Forest Feature Importances')
axes[1].set_ylabel('Mean impurity reduction')
axes[1].tick_params(axis='x', rotation=0)
axes[1].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('random_forest.png', dpi=150, bbox_inches='tight')
plt.show()

