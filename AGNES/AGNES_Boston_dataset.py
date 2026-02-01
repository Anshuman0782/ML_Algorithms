import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# -----------------------------
# Load Boston Housing Dataset
# -----------------------------
boston = fetch_openml(name="boston", version=1, as_frame=True)

X = boston.data   # features only (no target)

# -----------------------------
# OPTIONAL: take subset for dendrogram clarity
# -----------------------------
X = X.sample(n=120, random_state=42)

# -----------------------------
# Standardize Data
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# PCA for 2D Visualization
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# -----------------------------
# AGNES Function
# -----------------------------
def run_agnes(linkage_type):

    model = AgglomerativeClustering(
        n_clusters=3,
        linkage=linkage_type
    )

    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    plt.figure(figsize=(7, 5))

    for i in range(3):
        plt.scatter(
            X_pca[labels == i, 0],
            X_pca[labels == i, 1],
            label=f"Cluster {i+1}"
        )

    plt.title(f"AGNES — {linkage_type.capitalize()} Linkage\nSilhouette = {score:.3f}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()


# -----------------------------
# Run Required Linkages
# -----------------------------
run_agnes("single")
run_agnes("complete")
run_agnes("average")

# -----------------------------
# Dendrogram Function
# -----------------------------
def plot_dendrogram(method):

    Z = linkage(X_scaled, method=method)

    plt.figure(figsize=(12, 6))
    dendrogram(Z)
    plt.title(f"Dendrogram — {method.capitalize()} Linkage (Boston Housing)")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.show()


# -----------------------------
# Plot Required Dendrograms
# -----------------------------
plot_dendrogram("single")
plot_dendrogram("complete")
plot_dendrogram("average")
