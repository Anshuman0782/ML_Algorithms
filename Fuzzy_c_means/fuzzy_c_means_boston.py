import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import skfuzzy as fuzz

# -----------------------------
# Step 1: Load Boston Housing Dataset
# -----------------------------
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame

print("Dataset shape:", df.shape)

# Remove target column (MEDV) for clustering
X = df.drop("MEDV", axis=1)

# -----------------------------
# Step 2: Standardize data
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fuzzy C-means expects features as rows → transpose
data = X_scaled.T

# -----------------------------
# Step 3: Apply Fuzzy C-Means
# -----------------------------
n_clusters = 3

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data,
    c=n_clusters,
    m=2,              # fuzziness parameter
    error=0.005,
    maxiter=1000
)

# -----------------------------
# Step 4: Hard cluster labels
# -----------------------------
labels = np.argmax(u, axis=0)

print("\nCluster Centers:\n", cntr)
print("\nFuzzy Partition Coefficient (FPC):", fpc)

# -----------------------------
# Step 5: PCA for visualization
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# -----------------------------
# Step 6: Plot clusters
# -----------------------------
plt.figure(figsize=(8, 6))

for i in range(n_clusters):
    plt.scatter(
        X_pca[labels == i, 0],
        X_pca[labels == i, 1],
        label=f"Cluster {i+1}"
    )

plt.title("Fuzzy C-Means Clustering — Boston Housing")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# Step 7: Show membership values
# -----------------------------
print("\nSample membership values (first 5 rows):\n")
print(u[:, :5])
