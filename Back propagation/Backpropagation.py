import numpy as np

np.random.seed(0)

# -----------------------------
# Sample Input and Target
# -----------------------------
X = np.array([[1, 1, 0, 1]])     # input
T = np.array([[1, 0]])           # target

lr = 0.5

# -----------------------------
# Sigmoid Functions
# -----------------------------
def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(y):
    return y*(1-y)

# -----------------------------
# Initialize Weights
# -----------------------------
W1 = np.random.randn(4,3)   # 4→3
W2 = np.random.randn(3,2)   # 3→2
W3 = np.random.randn(2,2)   # 2→2

b1 = np.zeros((1,3))
b2 = np.zeros((1,2))
b3 = np.zeros((1,2))

# =============================
# FORWARD PASS
# =============================

Z1 = X @ W1 + b1
A1 = sigmoid(Z1)

Z2 = A1 @ W2 + b2
A2 = sigmoid(Z2)

Z3 = A2 @ W3 + b3
Y  = sigmoid(Z3)

print("Hidden1 Output:", A1)
print("Hidden2 Output:", A2)
print("Network Output:", Y)

# =============================
# ERROR
# =============================

E = T - Y
print("\nOutput Error:", E)

# =============================
# BACKPROPAGATION
# =============================

d3 = E * dsigmoid(Y)
d2 = (d3 @ W3.T) * dsigmoid(A2)
d1 = (d2 @ W2.T) * dsigmoid(A1)

# =============================
# WEIGHT UPDATES
# =============================

W3 += A2.T @ d3 * lr
b3 += d3 * lr

W2 += A1.T @ d2 * lr
b2 += d2 * lr

W1 += X.T @ d1 * lr
b1 += d1 * lr

print("\nUpdated W3:\n", W3)
print("\nUpdated W2:\n", W2)
print("\nUpdated W1:\n", W1)
