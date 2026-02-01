import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

# -----------------------------
# Step 1: Load Iris dataset
# -----------------------------
iris = datasets.load_iris()
X = iris.data

# -----------------------------
# Step 2: Standardize the data
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Step 3: Function to apply DIANA-like clustering
# -----------------------------
def apply_diana(linkage_type):
    Z = linkage(X_scaled, method=linkage_type)

    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title(f"DIANA Clustering ({linkage_type.capitalize()} Linkage)")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.show()

# -----------------------------
# Step 4: Apply different linkage methods
# -----------------------------
apply_diana("single")
apply_diana("complete")
apply_diana("average")
