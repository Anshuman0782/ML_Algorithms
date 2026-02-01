import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import skfuzzy as fuzz

# -----------------------------
# Step 1: Load MTCARS dataset
# -----------------------------
df = pd.read_csv("mtcars_sample.csv")

print("Dataset Preview:\n", df.head())

# -----------------------------
# Step 2: Remove non-numeric column if present
# -----------------------------
for col in df.columns:
    if df[col].dtype == "object":
        df = df.drop(col, axis=1)

# -----------------------------
# Step 3: Standardize data
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# FCM expects features as rows
data = X_scaled.T

# -----------------------------
# Step 4: Apply Fuzzy C-Means
# -----------------------------
n_clusters = 3

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data,
    c=n_clusters,
    m=2,
    error=0.005,
    maxiter=1000
)

# -----------------------------
# Step 5: Hard labels from fuzzy
# -----------------------------
labels = np.argmax(u, axis=0)

print("\nCluster Centers:\n", cntr)
print("\nFuzzy Partition Coefficient:", fpc)

# -----------------------------
# Step 6: PCA for visualization
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# -----------------------------
# Step 7: Plot clusters
# -----------------------------
plt.figure(figsize=(8, 6))

for i in range(n_clusters):
    plt.scatter(
        X_pca[labels == i, 0],
        X_pca[labels == i, 1],
        label=f"Cluster {i+1}"
    )

plt.title("Fuzzy C-Means Clustering — MTCARS Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# Step 8: Show membership values
# -----------------------------
print("\nMembership values (first 5 samples):\n")
print(u[:, :5])
