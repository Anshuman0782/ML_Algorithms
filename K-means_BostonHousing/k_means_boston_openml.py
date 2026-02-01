import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml

# -----------------------------
# Step 1: Load Boston Housing dataset (OpenML)
# -----------------------------
boston = fetch_openml(name="boston", version=1, as_frame=True)

df = boston.frame   # includes features + target

print("Dataset shape:", df.shape)
print(df.head())

# -----------------------------
# Step 2: Remove target column for clustering
# -----------------------------
# target column name = MEDV
X = df.drop("MEDV", axis=1)

# -----------------------------
# Step 3: Standardize the data
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Step 4: Apply K-Means clustering
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

labels = kmeans.labels_

# -----------------------------
# Step 5: Print results
# -----------------------------
print("\nCluster Centers:\n", kmeans.cluster_centers_)
print("\nInertia:", kmeans.inertia_)

# -----------------------------
# Step 6: PCA for visualization
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(kmeans.cluster_centers_)

# -----------------------------
# Step 7: Plot clusters
# -----------------------------
plt.figure(figsize=(8, 6))

for i in range(3):
    plt.scatter(
        X_pca[labels == i, 0],
        X_pca[labels == i, 1],
        label=f"Cluster {i+1}"
    )

plt.scatter(
    centers_pca[:, 0],
    centers_pca[:, 1],
    marker='x',
    s=200,
    label="Centroids"
)

plt.title("K-Means Clustering on Boston Housing (OpenML)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()
