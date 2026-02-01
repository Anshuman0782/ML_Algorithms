import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# -----------------------------
# Logic Gate Data
# -----------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_and = np.array([0, 0, 0, 1])
y_or  = np.array([0, 1, 1, 1])

# -----------------------------
# Train Perceptron — AND Gate
# -----------------------------
and_model = Perceptron(max_iter=1000, eta0=0.1, random_state=0)
and_model.fit(X, y_and)

# -----------------------------
# Train Perceptron — OR Gate
# -----------------------------
or_model = Perceptron(max_iter=1000, eta0=0.1, random_state=0)
or_model.fit(X, y_or)

# -----------------------------
# Predictions
# -----------------------------
print("AND Gate Predictions:")
for inp, pred in zip(X, and_model.predict(X)):
    print(inp, "→", pred)

print("\nOR Gate Predictions:")
for inp, pred in zip(X, or_model.predict(X)):
    print(inp, "→", pred)

# -----------------------------
# Function to Plot Decision Boundary
# -----------------------------
def plot_boundary(model, title):

    w = model.coef_[0]
    b = model.intercept_[0]

    x_vals = np.linspace(-0.5, 1.5, 100)
    y_vals = -(w[0]*x_vals + b) / w[1]

    plt.plot(x_vals, y_vals, label="Decision Boundary")


# -----------------------------
# Plot AND Gate
# -----------------------------
plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c=y_and, s=200)

plot_boundary(and_model, "AND")

plt.title("Perceptron — AND Gate")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.grid()
plt.show()

# -----------------------------
# Plot OR Gate
# -----------------------------
plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c=y_or, s=200)

plot_boundary(or_model, "OR")

plt.title("Perceptron — OR Gate")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.grid()
plt.show()

# -----------------------------
# Print Model Parameters
# -----------------------------
print("\nAND Weights:", and_model.coef_)
print("AND Bias:", and_model.intercept_)

print("\nOR Weights:", or_model.coef_)
print("OR Bias:", or_model.intercept_)