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



----------------------------------------------------------------------------------------------

🔹 Step 1 — Import Libraries
import numpy as np
import matplotlib.pyplot as plt

✅ Meaning

numpy → vector math

matplotlib → spatial visualization plots

🔹 Step 2 — Bipolar Input Matrix
X = np.array([
    [-1, -1],
    [-1,  1],
    [ 1, -1],
    [ 1,  1]
])

✅ Meaning

These are all combinations of bipolar inputs.

Truth table inputs:

x1	x2
-1	-1
-1	+1
+1	-1
+1	+1

Same inputs used for all gates.

🔹 Step 3 — Bipolar Target Outputs
y_and  = [-1, -1, -1,  1]
y_or   = [-1,  1,  1,  1]
y_nand = [ 1,  1,  1, -1]

✅ Meaning

These are bipolar truth tables.

Bipolar AND
x1	x2	AND
-1	-1	-1
-1	+1	-1
+1	-1	-1
+1	+1	+1

Same idea for OR and NAND.

🔹 Step 4 — Training Function
def train_perceptron(X, y, lr=1, epochs=20):

✅ Meaning

We create our own perceptron training instead of sklearn — this is better for viva because it shows learning rule.

Parameters:

X → inputs

y → target outputs

lr → learning rate

epochs → max training passes

🔹 Step 5 — Initialize Weights & Bias
w = np.zeros(2)
b = 0

✅ Meaning

Start with:

weights = [0, 0]
bias = 0


Perceptron will learn these.

🔹 Step 6 — Epoch Loop
for epoch in range(epochs):

✅ Meaning

Training repeats multiple passes over data.

Each pass = one epoch

🔹 Step 7 — Loop Through Each Training Sample
for xi, target in zip(X, y):

✅ Meaning

Take:

xi = input vector
target = expected output


Example:

xi = [-1, 1]
target = -1

🔹 Step 8 — Net Input Calculation
net = np.dot(w, xi) + b

✅ Formula

This is perceptron equation:

net = w1*x1 + w2*x2 + b

🔹 Step 9 — Activation Function
pred = 1 if net >= 0 else -1

✅ Meaning

Step function:

net ≥ 0 → +1
net < 0 → -1


This is bipolar activation.

🔹 Step 10 — Learning Rule (Most Important)
if pred != target:
    w = w + lr * target * xi
    b = b + lr * target

✅ This is THE perceptron update rule
Formula:
w_new = w_old + η * target * x
b_new = b_old + η * target


Where:

η = learning rate

x = input

target = correct label

Only update when prediction is wrong.

🧠 Example Update

If:

target = +1
xi = [1, -1]
lr = 1


Update:

w = w + [1, -1]
b = b + 1

🔹 Step 11 — Stop Early If No Errors
if error_count == 0:
    break

✅ Meaning

If all predictions correct → stop training early.

Efficient training.

🔹 Step 12 — Train Three Gates
w_and, b_and = train_perceptron(...)
w_or, b_or = train_perceptron(...)
w_nand, b_nand = train_perceptron(...)

✅ Meaning

Train three separate perceptrons.

Each learns different weights.

🔹 Step 13 — Prediction Function
def predict(X, w, b):


Computes:

net = w·x + b
apply step function


Used to verify results.

🔹 Step 14 — Plot Decision Boundary
y_vals = -(w[0]*x_vals + b) / w[1]

✅ Meaning

Convert perceptron equation into line:

w1*x + w2*y + b = 0
→ y = -(w1*x + b)/w2


This line separates classes.

📈 Plot Shows

Points = inputs

Colors = class (+1 / -1)

Line = learned boundary

This gives spatial understanding.
