import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# -----------------------------
# Load Dataset from CSV
# -----------------------------
df = pd.read_csv("mtcars.csv")

print("Dataset Preview:\n", df.head())

# -----------------------------
# Remove non-numeric column if present
# -----------------------------
# Some mtcars CSV files contain car names column
for col in df.columns:
    if df[col].dtype == "object":
        df = df.drop(col, axis=1)

# -----------------------------
# Standardize Features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# -----------------------------
# Apply AGNES Clustering
# -----------------------------
model = AgglomerativeClustering(
    n_clusters=3,
    linkage="average"   # you can change: single / complete / average
)

labels = model.fit_predict(X_scaled)

# -----------------------------
# Silhouette Score
# -----------------------------
score = silhouette_score(X_scaled, labels)
print("\nSilhouette Score:", score)

# -----------------------------
# PCA for Visualization
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# -----------------------------
# Plot Cluster Visualization
# -----------------------------
plt.figure(figsize=(8, 6))

for i in range(3):
    plt.scatter(
        X_pca[labels == i, 0],
        X_pca[labels == i, 1],
        label=f"Cluster {i+1}"
    )

plt.title("AGNES Clustering — MTCARS Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

# -----------------------------
# Dendrogram (Bonus — if asked)
# -----------------------------
Z = linkage(X_scaled, method="average")

plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.title("Dendrogram — AGNES (Average Linkage) — MTCARS")
plt.xlabel("Cars")
plt.ylabel("Distance")
plt.show()
