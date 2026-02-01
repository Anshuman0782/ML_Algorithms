import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids

# -----------------------------
# Step 1: Load Iris Dataset
# -----------------------------
iris = load_iris()
X = iris.data
y_true = iris.target

print("Dataset shape:", X.shape)
print("Classes:", iris.target_names)

# -----------------------------
# Step 2: Standardize Features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Step 3: Apply K-Medoids
# -----------------------------
kmedoids = KMedoids(
    n_clusters=3,
    metric="euclidean",
    random_state=42
)

labels = kmedoids.fit_predict(X_scaled)

print("\nMedoid indices:", kmedoids.medoid_indices_)
print("\nMedoid points:\n", kmedoids.cluster_centers_)

# -----------------------------
# Step 4: Cluster Quality
# -----------------------------
score = silhouette_score(X_scaled, labels)
print("\nSilhouette Score:", round(score, 3))

# -----------------------------
# Step 5: PCA for Spatial Plot
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
medoids_pca = pca.transform(kmedoids.cluster_centers_)

# -----------------------------
# Step 6: Plot Clusters + Medoids
# -----------------------------
plt.figure(figsize=(8,6))

for i in range(3):
    plt.scatter(
        X_pca[labels == i, 0],
        X_pca[labels == i, 1],
        label=f"Cluster {i+1}"
    )

# plot medoids clearly
plt.scatter(
    medoids_pca[:,0],
    medoids_pca[:,1],
    marker='X',
    s=300,
    label="Medoids"
)

plt.title("K-Medoids Clustering — Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()