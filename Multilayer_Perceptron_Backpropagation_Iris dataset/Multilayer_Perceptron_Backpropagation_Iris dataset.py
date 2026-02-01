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





-----------------------------------------------------------------------


🔹 Step 1 — Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

✅ Meaning

load_iris → get Iris dataset

MLPClassifier → multilayer perceptron (uses backprop)

train_test_split → divide data

metrics → measure performance

matplotlib → draw network diagram

🔹 Step 2 — Load Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target

✅ Meaning

Iris dataset contains:

X → features (150 × 4)
y → class labels (0,1,2)


Features:

sepal length

sepal width

petal length

petal width

Classes:

setosa, versicolor, virginica

🔹 Step 3 — Train/Test Split
X_train, X_test, y_train, y_test =
    train_test_split(X, y, test_size=0.25, random_state=42)

✅ Meaning

Split data:

75% → training
25% → testing


Why?

👉 Train on one part
👉 Test on unseen data

random_state keeps split same each run.

🔹 Step 4 — Create MLP Model
mlp = MLPClassifier(
    hidden_layer_sizes=(6,4),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=1
)


This defines your neural network.

🧩 Network Architecture
Input layer = 4 neurons   (Iris features)
Hidden layer 1 = 6 neurons
Hidden layer 2 = 4 neurons
Output layer = 3 neurons  (3 classes)

⚙️ Parameters Explained
hidden_layer_sizes=(6,4)

Two hidden layers:

Layer1 → 6 neurons
Layer2 → 4 neurons

activation='relu'

Activation function used in hidden layers:

ReLU = max(0,x)


Adds nonlinearity.

solver='adam'

Optimization algorithm for backprop:

Adaptive gradient descent


Fast + stable.

max_iter=2000

Maximum training iterations (epochs).

Ensures convergence.

🔹 Step 5 — Train Model (Backprop Happens Here)
mlp.fit(X_train, y_train)

✅ What Happens Internally

This runs:

forward pass
compute error
backpropagate gradients
update weights
repeat many epochs


You don’t see math — sklearn handles it.

But this is true backpropagation training.

🔹 Step 6 — Predictions
y_pred = mlp.predict(X_test)

✅ Meaning

Model predicts flower class for test data.

🔹 Step 7 — Accuracy & Report
accuracy_score(...)
classification_report(...)

✅ Output Shows
Accuracy %
Precision
Recall
F1 score
per class performance


Good for lab record.

🔹 Step 8 — Show Network Details
print(mlp.n_features_in_)
print(mlp.hidden_layer_sizes)
print(mlp.n_outputs_)

✅ Meaning

Prints learned network structure:

Input neurons = 4
Hidden = (6,4)
Output neurons = 3


Confirms topology.

🔹 Step 9 — Draw Network Diagram Function
def draw_network(layer_sizes):


This is a visual helper, not training.

It draws:

nodes = neurons
lines = connections

Inside Drawing Logic
Loop layers
for each layer:
    draw neurons as circles

Connect layers
draw lines between every neuron


Because MLP is fully connected.

🔹 Step 10 — Call Diagram
layer_sizes = [4, 6, 4, 3]
draw_network(layer_sizes)

✅ Meaning

Matches your trained MLP:

4 → 6 → 4 → 3


Plot appears showing network structure.
