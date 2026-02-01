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





---------------------------------------------------------------------------------------------


Step 1 — Imports & Seed
import numpy as np
np.random.seed(0)

✅ Meaning

numpy → matrix math

random seed → same weights every run (repeatable output for lab)

🔹 Step 2 — Input & Target
X = np.array([[1, 1, 0, 1]])
T = np.array([[1, 0]])

✅ Meaning

X = one training sample

T = desired output class label

Shape:

X → (1 × 4)
T → (1 × 2)

🔹 Step 3 — Learning Rate
lr = 0.5

✅ Meaning

Controls how much weights change during update.

Higher → faster but risky
Lower → slower but stable

🔹 Step 4 — Sigmoid Functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(y):
    return y*(1-y)

✅ Meaning
Sigmoid activation

Converts net input → value between 0 and 1.

Derivative

Needed for backprop gradient:

σ’ = y(1−y)


We pass output y directly — faster.

🔹 Step 5 — Weight Initialization
W1 = np.random.randn(4,3)
W2 = np.random.randn(3,2)
W3 = np.random.randn(2,2)

✅ Meaning

Weight matrices match layer sizes:

From	To	Shape
Input	Hidden1	4×3
Hidden1	Hidden2	3×2
Hidden2	Output	2×2
Biases
b1 = zeros(1×3)
b2 = zeros(1×2)
b3 = zeros(1×2)


Bias shifts neuron threshold.

=============================
🚀 FORWARD PASS
=============================

This computes prediction.

🔹 Hidden Layer 1
Z1 = X @ W1 + b1
A1 = sigmoid(Z1)

✅ Meaning
Z1 = weighted sum
A1 = activated output


Shape:

A1 → (1×3)

🔹 Hidden Layer 2
Z2 = A1 @ W2 + b2
A2 = sigmoid(Z2)


Output:

A2 → (1×2)

🔹 Output Layer
Z3 = A2 @ W3 + b3
Y  = sigmoid(Z3)

✅ Meaning

Final network prediction:

Y = predicted output (1×2)


Printed as:

Hidden1 Output
Hidden2 Output
Network Output

=============================
❌ ERROR
=============================
E = T - Y

✅ Meaning

Difference between:

target – prediction


This drives learning.

=============================
🔁 BACKPROPAGATION
=============================

This is the core algorithm.

We compute error signals (deltas) backward.

🔹 Output Layer Delta
d3 = E * dsigmoid(Y)

Formula
delta_output = error × sigmoid_derivative

🔹 Hidden Layer 2 Delta
d2 = (d3 @ W3.T) * dsigmoid(A2)

Meaning

Error flows backward:

next_delta × weights × derivative


Chain rule applied.

🔹 Hidden Layer 1 Delta
d1 = (d2 @ W2.T) * dsigmoid(A1)


Same idea — propagate further back.

=============================
🔧 WEIGHT UPDATES
=============================

Gradient descent step.

🔹 Update Output Weights
W3 += A2.T @ d3 * lr
b3 += d3 * lr


Formula:

weight += inputᵀ × delta × lr

🔹 Update Hidden Weights

Same pattern:

W2 += A1.T @ d2 * lr
W1 += X.T  @ d1 * lr


Each layer uses:

previous layer output × current delta

📊 What You See Printed

The code prints:

✅ Layer outputs
Hidden1 Output
Hidden2 Output
Network Output

✅ Error vector
Output Error

✅ Updated weights
W1, W2, W3 after learning


That proves backprop worked.
