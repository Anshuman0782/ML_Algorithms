import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# -----------------------------
# Load Dataset
# -----------------------------
iris = datasets.load_iris()
X = iris.data

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
# Function: Run AGNES + Plot
# -----------------------------
def run_agnes(linkage_type):

    model = AgglomerativeClustering(
        n_clusters=3,
        linkage=linkage_type
    )

    labels = model.fit_predict(X_scaled)

    # Silhouette Score
    score = silhouette_score(X_scaled, labels)

    # Plot
    plt.figure(figsize=(7, 5))
    for i in range(3):
        plt.scatter(
            X_pca[labels == i, 0],
            X_pca[labels == i, 1],
            label=f"Cluster {i+1}"
        )

    plt.title(f"AGNES ({linkage_type.capitalize()} Linkage)\nSilhouette Score = {score:.3f}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()


# -----------------------------
# Run All Linkage Methods
# -----------------------------
run_agnes("single")
run_agnes("complete")
run_agnes("average")
run_agnes("ward")


# -----------------------------
# Function: Plot Dendrogram
# -----------------------------
def plot_dendrogram(method_name):

    Z = linkage(X_scaled, method=method_name)

    plt.figure(figsize=(12, 6))
    dendrogram(Z)
    plt.title(f"Dendrogram (AGNES - {method_name.capitalize()} Linkage)")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.show()


# -----------------------------
# Plot Dendrograms
# -----------------------------
plot_dendrogram("single")
plot_dendrogram("complete")
plot_dendrogram("average")
plot_dendrogram("ward")
