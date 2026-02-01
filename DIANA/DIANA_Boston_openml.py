import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

# -----------------------------
# Step 1: Load Boston Housing (OpenML)
# -----------------------------
boston = fetch_openml(name="boston", version=1, as_frame=True)

df = boston.frame
print("Dataset shape:", df.shape)

# Remove target column MEDV
X = df.drop("MEDV", axis=1)

# -----------------------------
# Optional: sample for clear dendrogram
# -----------------------------
X = X.sample(n=120, random_state=42)

# -----------------------------
# Step 2: Standardize data
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Step 3: Dendrogram Function
# -----------------------------
def apply_diana_like(linkage_type):

    Z = linkage(X_scaled, method=linkage_type)

    plt.figure(figsize=(11, 7))
    dendrogram(Z, leaf_rotation=90)
    plt.title(f"Hierarchical Clustering ({linkage_type.capitalize()} Linkage)\nBoston Housing — OpenML")
    plt.xlabel("Housing Samples")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Step 4: Apply Linkages
# -----------------------------
apply_diana_like("single")
apply_diana_like("complete")
apply_diana_like("average")
