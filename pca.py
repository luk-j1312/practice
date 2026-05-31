import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.random.seed(42)
n=1000
# expand the feature set so that PCA has more with which to work
hour = np.random.randint(0,24,n)
temp        = np.random.normal(75, 15, n)
load        = np.random.normal(50, 10, n)
wind        = np.random.normal(30, 10, n)
solar       = np.clip(np.random.normal(20, 8, n), 0, None)
natural_gas = np.random.normal(3.5, 0.5, n)   # gas price ($/MMBtu)
# introduce correlations that would appear in real data:
# - high temp drives higher load (cooling demand)
# - high load drives higher gas prices (more generation needed)
load        = load + 0.4 * temp
natural_gas = natural_gas + 0.02 * load
peak_hour   = ((hour >= 15) & (hour <= 20)).astype(float)
df = pd.DataFrame({
    'temp':        temp,
    'load':        load,
    'wind':        wind,
    'solar':       solar,
    'natural_gas': natural_gas,
    'peak_hour':   peak_hour
})

print("Feature correlations:")
print(df.corr().round(2))
print()

# ---standardize---
scaler = StandardScaler()
X_sc = scaler.fit_transform(df)
# ---fit PCA (keep all components first)---
pca = PCA()
pca.fit(X_sc)
# explained variance
evr = pca.explained_variance_ratio_
cumulative = np.cumsum(evr)
print("Explained variance by component:")
print("-" * 45)
for i, (ev, cum) in enumerate(zip(evr, cumulative)):
    print(f"PC{i+1}:  {ev:.3f}  ({ev*100:.1f}%)   "
          f"cumulative: {cum*100:.1f}%")

# ---loadings: how much each feature contributes to each PC
loadings = pd.DataFrame(pca.components_.T,   # transpose: rows=features, cols=PCs
    index=df.columns, columns=[f'PC{i+1}' for i in range(len(df.columns))])
print("\nLoadings (eigenvectors):")
print(loadings.round(3))

# ---project data onto first 2 PCs---
pca_2 = PCA(n_components=2)
X_reduced = pca_2.fit_transform(X_sc)
print(f"\nOriginal shape: {X_sc.shape}")
print(f"Reduced shape:  {X_reduced.shape}")
print(f"Variance retained: "
      f"{pca_2.explained_variance_ratio_.sum()*100:.1f}%")

# ---plots---
fig, axes = plt.subplots(1,3,figsize=(16,5))
# Scree plot
axes[0].bar(range(1, len(evr)+1), evr*100, color='steelblue', edgecolor='white', 
            label='Individual')
axes[0].plot(range(1, len(evr)+1), cumulative*100, 'o-', color='tomato', label='Cumulative')
axes[0].axhline(90, color='black', linestyle='--', lw=1, label='90% threshold')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained (%)')
axes[0].set_title('Scree Plot')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
# Loading heatmap
im = axes[1].imshow(loadings.values, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
axes[1].set_xticks(range(len(df.columns)))
axes[1].set_xticklabels([f'PC{i+1}' for i in range(len(df.columns))])
axes[1].set_yticks(range(len(df.columns)))
axes[1].set_yticklabels(df.columns)
axes[1].set_title('Loadings Heatmap')
plt.colorbar(im, ax=axes[1])
# Scatter plot of first two PCs
axes[2].scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.3, s=10, color='steelblue')
axes[2].set_xlabel(f'PC1 ({evr[0]*100:.1f}% variance)')
axes[2].set_ylabel(f'PC2 ({evr[1]*100:.1f}% variance)')
axes[2].set_title('Data Projected onto First Two PCs')
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca.png', dpi=150, bbox_inches='tight')
plt.show()

