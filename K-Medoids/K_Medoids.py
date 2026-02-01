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




-------------------------------------------------------------------------




🔹 Step 1 — Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids

✅ Meaning

load_iris → built-in Iris dataset

StandardScaler → normalize features

KMedoids → clustering algorithm

PCA → reduce dimensions for plotting

silhouette_score → cluster quality measure

matplotlib → visualization

⚠️ KMedoids comes from sklearn-extra, not core sklearn.

🔹 Step 2 — Load Dataset
iris = load_iris()
X = iris.data
y_true = iris.target

✅ Meaning
X = 150 rows × 4 features


Features:

sepal length

sepal width

petal length

petal width

y_true = real species labels (not used in clustering — only for reference)

🔹 Step 3 — Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

✅ Why Needed

Clustering uses distance.

Features have different scales:

cm vs ratios vs counts


Standardization makes:

mean = 0
std = 1


So no feature dominates distance.

🔹 Step 4 — Apply K-Medoids
kmedoids = KMedoids(
    n_clusters=3,
    metric="euclidean",
    random_state=42
)

✅ Meaning

We configure the model:

n_clusters=3 → Iris has 3 natural groups

metric → distance type

random_state → repeatable result

Train + Get Labels
labels = kmedoids.fit_predict(X_scaled)

✅ Meaning

Algorithm:

1️⃣ Pick initial medoids
2️⃣ Assign points to nearest medoid
3️⃣ Swap medoid candidates
4️⃣ Minimize total distance
5️⃣ Repeat until stable

Output:

labels → cluster number for each sample

🔹 Step 5 — Print Medoids
kmedoids.medoid_indices_
kmedoids.cluster_centers_

✅ Meaning

Unlike K-Means:

center = mean (not real point)


K-Medoids:

center = actual data point


These are the chosen representative samples.

🔹 Step 6 — Cluster Quality Score
score = silhouette_score(X_scaled, labels)

✅ Meaning

Silhouette score measures:

how well points fit their cluster


Range:

-1 → bad
0 → overlap
+1 → good separation


Higher = better clustering.

🔹 Step 7 — PCA for Spatial Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

✅ Why Needed

Data is 4-dimensional → cannot plot directly.

PCA converts:

4D → 2D


while preserving most variance.

This allows spatial plotting.

Transform Medoids Too
medoids_pca = pca.transform(kmedoids.cluster_centers_)


So medoids appear in same 2D space.

🔹 Step 8 — Plot Clusters
for i in range(3):
    plt.scatter(X_pca[labels == i, 0],
                X_pca[labels == i, 1])

✅ Meaning

For each cluster:

plot its points with same color


Creates visual grouping.

🔹 Step 9 — Plot Medoids
plt.scatter(medoids_pca[:,0],
            medoids_pca[:,1],
            marker='X',
            s=300)

✅ Meaning

Medoids are shown as:

big X markers
larger size
different symbol


So they stand out clearly.

This gives strong spatial understanding.

🔹 Step 10 — Labels & Grid
plt.title(...)
plt.xlabel(...)
plt.ylabel(...)
plt.legend()
plt.grid()

✅ Meaning

Makes plot readable and lab-presentable.

🎯 Final Output You See
📊 Console
Medoid indices
Medoid values
Silhouette score

📈 Plot
Colored clusters
Separated groups
Big X = medoids


This is the spatial output your question asked for.
plt.show()
