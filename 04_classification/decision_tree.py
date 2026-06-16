import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

np.random.seed(42)
n=1000

hour = np.random.randint(0, 24, n)
temp = np.random.normal(75, 15, n)
load = np.random.normal(50, 10, n)
wind = np.random.normal(30, 10, n)

log_odds = (-4 + 0.05*load - 0.08*wind + 0.03*temp + 
            0.15*((hour >= 15) & (hour <= 20)).astype(float))
prob_spike = 1 / (1 + np.exp(-log_odds))
spike = (np.random.uniform(0, 1, n) < prob_spike).astype(int)

df = pd.DataFrame({"temp": temp, "load": load, "wind": wind, "peak_hour": ((hour >= 15) & (hour <= 20)).astype(int), "spike": spike})
X = df.drop("spike", axis=1)
Y = df["spike"]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

# Note: trees don't require scaling---splits are threshold-based, not distance-based. 
# We skip StandardScaler here intentionally.

# --- Depth experiment: bias-variance tradeoff in action ---
depths = range(1, 15)
train_aucs, cv_aucs = [], []

for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    tree.fit(X_train, Y_train)

    # Training AUC
    train_prob = tree.predict_proba(X_train)[:, 1]
    train_aucs.append(roc_auc_score(Y_train, train_prob))

    # CV AUC
    cv_scores = cross_val_score(tree, X_train, Y_train,
                                cv=5, scoring='roc_auc')
    cv_aucs.append(cv_scores.mean())

best_depth = depths[np.argmax(cv_aucs)]
print(f"best depth by CV AUC: {best_depth}")
print(f"CV AUC at best depth: {max(cv_aucs):.3f}")

# --- Fit final tree at best depth ---
best_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
best_tree.fit(X_train, Y_train)
test_prob = best_tree.predict_proba(X_test)[:, 1]
print(f"Test AUC: {roc_auc_score(Y_test, test_prob):.3f}")
# --- Feature importance ---
importance = pd.Series(best_tree.feature_importances_, 
                       index=X.columns).sort_values(ascending=False)
print(f"\nFeature importances:\n{importance.round(3)}")

# --- Plots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Bias-variance curve
axes[0].plot(depths, train_aucs, 'o-', color='tomato', label='Training AUC')
axes[0].plot(depths, cv_aucs, 'o-', color='steelblue', label='CV AUC')
axes[0].axvline(best_depth, color='black', linestyle='--', lw=1, 
                label=f'Best depth = {best_depth}')
axes[0].set_xlabel('Max Depth')
axes[0].set_ylabel('AUC')
axes[0].set_title('Bias-Variance Tradeoff in Decision Tree')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
# Feature importance
importance.plot(kind='bar', ax=axes[1], color='steelblue', edgecolor='white')
axes[1].set_title('Feature Importances')
axes[1].set_ylabel('Importance (mean impurity reduction)')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=0)
axes[1].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Visualize the actual tree structure ---
fig2, ax2 = plt.subplots(figsize=(16, 6))
plot_tree(best_tree, feature_names=X.columns.tolist(),
          class_names=['No Spike', 'Spike'], filled=True, rounded=True, 
          fontsize=9, ax=ax2)
plt.title(f'Decision Tree (max_depth={best_depth})')
plt.tight_layout()
plt.savefig('tree_structure.png', dpi=150, bbox_inches='tight')
plt.show()