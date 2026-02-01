import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Step 1: Load Iris Dataset
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target

print("Feature shape:", X.shape)
print("Classes:", iris.target_names)

# -----------------------------
# Step 2: Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

# -----------------------------
# Step 3: Train MLP (Backprop)
# -----------------------------
mlp = MLPClassifier(
    hidden_layer_sizes=(6,4),   # two hidden layers
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=1
)

mlp.fit(X_train, y_train)

# -----------------------------
# Step 4: Predictions
# -----------------------------
y_pred = mlp.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred))

# -----------------------------
# Step 5: Show Network Details
# -----------------------------
print("\nLayer sizes:")
print("Input:", mlp.n_features_in_)
print("Hidden layers:", mlp.hidden_layer_sizes)
print("Output:", mlp.n_outputs_)

# -----------------------------
# Step 6: Draw Neural Network Diagram
# -----------------------------
layer_sizes = [4, 6, 4, 3]   # input, hidden1, hidden2, output

def draw_network(layer_sizes):
    fig, ax = plt.subplots(figsize=(8,6))
    v_spacing = 1
    h_spacing = 2

    for i, size in enumerate(layer_sizes):
        for j in range(size):
            ax.scatter(i*h_spacing, j*v_spacing, s=800)

            if i > 0:
                for k in range(layer_sizes[i-1]):
                    ax.plot(
                        [(i-1)*h_spacing, i*h_spacing],
                        [k*v_spacing, j*v_spacing]
                    )

    ax.set_title("Trained MLP Network Structure (Iris)")
    ax.axis('off')
    plt.show()

draw_network(layer_sizes)
