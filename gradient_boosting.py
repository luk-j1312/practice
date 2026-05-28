from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
X=df.drop("spike", axis=1)
Y=df["spike"]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# --- learning rate experiment ---
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
print("Learning rate experiment (100 trees, depth=3)")
print("-"*45)
for lr in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=lr, 
                                    max_depth=3, random_state=42)
    scores = cross_val_score(gb, X_train, Y_train, cv=5, scoring='roc_auc')
    print(f"lr={lr:<6}  CV AUC={scores.mean():.3f}"
          f"  (+/- {scores.std():.3f})")
    
# --- Final model: fit and compare all methods ---
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, 
                            learning_rate=0.05, max_depth=3, random_state=42)
}
print("\nFinal model comparison:")
print("-"*45)
for name, model in models.items():
    cv_auc = cross_val_score(model, X_train, Y_train, cv=5, scoring='roc_auc').mean()
    model.fit(X_train, Y_train)
    test_auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])
    print(f"{name:<25} CV={cv_auc:.3f}  Test={test_auc:.3f}")

# --- Boosting: training vs. CV AUC across n_estimators ---
gb_final = GradientBoostingClassifier(learning_rate=0.05, max_depth=3, random_state=42,
                                        n_estimators=500)
gb_final.fit(X_train, Y_train)
# staged_predict_proba gives predictions after each tree
train_aucs, test_aucs = [], []
for train_pred, test_pred in zip(gb_final.staged_predict_proba(X_train), 
                                 gb_final.staged_predict_proba(X_test)):
    train_aucs.append(roc_auc_score(Y_train, train_pred[:, 1]))
    test_aucs.append(roc_auc_score(Y_test,  test_pred[:, 1]))

# feature importance from gradient boosting
gb_importance = pd.Series(
    gb_final.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

fig, axes = plt.subplots(1,2,figsize=(13,5))
# AUC vs n_estimators
axes[0].plot(train_aucs, color='tomato', lw=1.5, label='Training AUC')
axes[0].plot(test_aucs,  color='steelblue', lw=1.5, label='Test AUC')
axes[0].axhline(0.747, color='green', linestyle='--', lw=1, 
                label='Logistic regression (0.747)')
axes[0].set_xlabel('Number of Trees')
axes[0].set_ylabel('AUC')
axes[0].set_title('Gradient Boosting: AUC vs. Trees (lr=0.05)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
# Feature importances
gb_importance.plot(kind='bar', ax=axes[1], color='steelblue', edgecolor='white')
axes[1].set_title('Gradient Boosting Feature Importances')
axes[1].set_ylabel('Relative importance')
axes[1].tick_params(axis='x', rotation=0)
axes[1].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('gradient_boosting.png', dpi=150, bbox_inches='tight')
plt.show()