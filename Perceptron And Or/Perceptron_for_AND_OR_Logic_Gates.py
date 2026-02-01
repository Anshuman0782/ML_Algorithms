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



----------------------------------------------------------------------------------

This program trains a Perceptron (single neuron) to learn:

AND gate
OR gate


and then shows the decision boundary visually.

🔹 Step 1 — Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

✅ Meaning

numpy → create input arrays

matplotlib → draw plots (spatial visualization)

Perceptron → built-in perceptron model from sklearn

🔹 Step 2 — Define Logic Gate Inputs
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

✅ Meaning

These are the 4 possible binary inputs:

Input1	Input2
0	0
0	1
1	0
1	1

This same input set is used for both AND and OR.

🔹 Step 3 — Define Outputs (Targets)
y_and = np.array([0, 0, 0, 1])
y_or  = np.array([0, 1, 1, 1])

✅ Meaning

These are the correct outputs:

AND Gate Truth Table
00 → 0
01 → 0
10 → 0
11 → 1

OR Gate Truth Table
00 → 0
01 → 1
10 → 1
11 → 1


These are what the perceptron must learn.

🔹 Step 4 — Train Perceptron for AND Gate
and_model = Perceptron(max_iter=1000, eta0=0.1, random_state=0)
and_model.fit(X, y_and)

✅ Meaning

We create and train a perceptron.

Parameters:

max_iter=1000 → max training passes

eta0=0.1 → learning rate

fit() → train using AND outputs

During training perceptron:

adjusts weights
adjusts bias
until outputs match targets

🔹 Step 5 — Train Perceptron for OR Gate
or_model = Perceptron(...)
or_model.fit(X, y_or)


Same process — but trained with OR truth table.

So now we have:

one perceptron for AND
one perceptron for OR

🔹 Step 6 — Make Predictions
for inp, pred in zip(X, and_model.predict(X)):
    print(inp, "→", pred)

✅ Meaning

We test the model on all inputs.

Example output:

[1 1] → 1


Means perceptron correctly learned the gate.

🔹 Step 7 — Decision Boundary Function
def plot_boundary(model, title):
    w = model.coef_[0]
    b = model.intercept_[0]

✅ Meaning

Perceptron learns equation:

w1*x1 + w2*x2 + b = 0


This is a line → decision boundary.

We extract:

weights (w)
bias (b)

Boundary Line Formula
y_vals = -(w[0]*x_vals + b) / w[1]


This converts perceptron equation into:

y = mx + c


So we can draw the separating line.

🔹 Step 8 — Plot AND Gate
plt.scatter(X[:,0], X[:,1], c=y_and, s=200)
plot_boundary(and_model, "AND")

✅ Meaning

Plot shows:

Points = inputs

Colors = class (0 or 1)

Line = perceptron boundary

This gives spatial understanding of how perceptron separates classes.

🔹 Step 9 — Plot OR Gate

Same process — different learned boundary line.

🔹 Step 10 — Print Weights & Bias
print(and_model.coef_)
print(and_model.intercept_)

✅ Meaning

Shows learned parameters:

weights → feature importance
bias → threshold shift


Perceptron decision rule:

if w·x + b ≥ 0 → class 1
else → class 0

🧠 What Perceptron Is Doing Internally

For each training sample:

prediction = sign(w·x + b)

if wrong:
    w = w + lr * x * error
    b = b + lr * error


Repeat until correct.

🎯 Why AND & OR Work

Because they are:

linearly separable


One straight line can separate classes.
