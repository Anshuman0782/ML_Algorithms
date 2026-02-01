import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Bipolar Inputs
# -----------------------------
X = np.array([
    [-1, -1],
    [-1,  1],
    [ 1, -1],
    [ 1,  1]
])

# -----------------------------
# Bipolar Targets
# -----------------------------
y_and  = np.array([-1, -1, -1,  1])
y_or   = np.array([-1,  1,  1,  1])
y_nand = np.array([ 1,  1,  1, -1])


# -----------------------------
# Perceptron Training Function
# -----------------------------
def train_perceptron(X, y, lr=1, epochs=20):

    w = np.zeros(2)
    b = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}")
        error_count = 0

        for xi, target in zip(X, y):

            net = np.dot(w, xi) + b
            pred = 1 if net >= 0 else -1

            if pred != target:
                w = w + lr * target * xi
                b = b + lr * target
                error_count += 1
                print(" Updated w:", w, " b:", b)

        if error_count == 0:
            break

    return w, b


# -----------------------------
# Train Models
# -----------------------------
print("\n=== Training Bipolar AND ===")
w_and, b_and = train_perceptron(X, y_and)

print("\n=== Training Bipolar OR ===")
w_or, b_or = train_perceptron(X, y_or)

print("\n=== Training Bipolar NAND ===")
w_nand, b_nand = train_perceptron(X, y_nand)


# -----------------------------
# Prediction Function
# -----------------------------
def predict(X, w, b):
    net = np.dot(X, w) + b
    return np.where(net >= 0, 1, -1)


# -----------------------------
# Show Results
# -----------------------------
print("\nAND Predictions:", predict(X, w_and, b_and))
print("OR Predictions:", predict(X, w_or, b_or))
print("NAND Predictions:", predict(X, w_nand, b_nand))


# -----------------------------
# Plot Decision Boundary
# -----------------------------
def plot_gate(X, y, w, b, title):

    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], c=y, s=200)

    x_vals = np.linspace(-2, 2, 100)
    y_vals = -(w[0]*x_vals + b) / w[1]

    plt.plot(x_vals, y_vals)

    plt.title(title)
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.grid()
    plt.show()


plot_gate(X, y_and, w_and, b_and, "Bipolar AND")
plot_gate(X, y_or, w_or, b_or, "Bipolar OR")
plot_gate(X, y_nand, w_nand, b_nand, "Bipolar NAND")