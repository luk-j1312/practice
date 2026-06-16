import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
np.random.seed(42)
n=1000
# Same dataset as PCA
hour        = np.random.randint(0, 24, n)
temp        = np.random.normal(75, 15, n)
load        = np.random.normal(50, 10, n)
wind        = np.random.normal(30, 10, n)
solar       = np.clip(np.random.normal(20, 8, n), 0, None)
natural_gas = np.random.normal(3.5, 0.5, n)
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
scaler = StandardScaler()
X_sc = scaler.fit_transform(df)

# ---elbow method: WCSS and silhouette score across k---
wcss = []
k_range = range(2, 11)
silhouettes = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_sc)
    wcss.append(kmeans.inertia_)   #  inertia_ is WCSS
    silhouettes.append(silhouette_score(X_sc, labels))
    print(f"k={k}  WCSS={kmeans.inertia_:7.1f}"
          f"  Silhouette={silhouettes[-1]:.3f}")

# ---fit final model at k=3 (I'll verify whether this matches the elbow/silhouette results)---
best_k = 3
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = kmeans_final.fit_predict(X_sc)
# ---cluster profiles: mean of each feature per cluster---
profile = df.groupby('cluster').mean().round(2)
print("\nCluster profiles:")
print(profile)

# project onto 2 PCs for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sc)

# ---plots---
fig, axes = plt.subplots(1,3,figsize=(16,5))
# Elbow plot
axes[0].plot(k_range, wcss, 'o-', color='steelblue')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('WCSS')
axes[0].set_title('Elbow Method')
axes[0].grid(True, alpha=0.3)
# Silhouette scores
axes[1].plot(k_range, silhouettes, 'o-', color='tomato')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score by k')
axes[1].grid(True, alpha=0.3)
# Cluster scatter on first 2 PCs
colors = ['steelblue', 'tomato', 'seagreen']
for c in range(best_k):
    mask = df['cluster'] == c
    axes[2].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    alpha=0.4, s=15, color=colors[c],
                    label=f'Cluster {c}')
axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[2].set_title('Clusters in PC Space (k=3)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans.png', dpi=150, bbox_inches='tight')
plt.show()