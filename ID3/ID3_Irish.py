import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Step 1: Load Iris Dataset
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target

feature_names = iris.feature_names
class_names = iris.target_names

print("Feature Names:", feature_names)
print("Class Names:", class_names)

# -----------------------------
# Step 2: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

# -----------------------------
# Step 3: ID3 = Decision Tree with Entropy
# -----------------------------
model = DecisionTreeClassifier(
    criterion="entropy",   # ID3
    max_depth=4,           # keeps tree readable
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Step 4: Predictions
# -----------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Step 5: Text Rules (Very Helpful Output)
# -----------------------------
print("\nDecision Tree Rules:\n")
print(export_text(model, feature_names=feature_names))

# -----------------------------
# Step 6: Draw Spatial Decision Tree
# -----------------------------
plt.figure(figsize=(18, 10))

plot_tree(
    model,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=11
)

plt.title("ID3 Decision Tree — Iris Dataset")
plt.show()





✅ 2️⃣ Accuracy = 1.0
Accuracy: 1.0

Meaning (simple)

Model predicted 100% test samples correctly

Accuracy formula:

correct predictions / total predictions


Here:

38 correct / 38 total = 1.0

⚠️ Viva Note

Iris dataset is very clean → decision trees often get perfect accuracy.

This is normal, not cheating.

✅ 3️⃣ Classification Report

Example line:

precision recall f1-score support


Let’s decode one row:

class 0 (setosa)
precision = 1.00
recall = 1.00
f1 = 1.00
support = 15

Meanings in easy words
🎯 Precision

When model predicts setosa → how often correct
= 100%

🔍 Recall

Out of all real setosa → how many found
= 100%

⚖️ F1 Score

Balance of precision & recall
= Perfect

📦 Support

Number of test samples in that class

✅ 4️⃣ Decision Tree Rules — Most Important Part

This is your ID3 decision logic:

petal length <= 2.45 → class 0

🌸 Rule 1 — Setosa Detection

If:

petal length ≤ 2.45


→ Always setosa

This is biologically true — setosa has very small petals.

So ID3 correctly chose petal length as root split
(because highest information gain).

🌸 Rule 2 — Versicolor vs Virginica Split

Next:

petal length ≤ 4.75
    petal width ≤ 1.65 → versicolor
    else → virginica


Meaning:

Medium petals → check width:

thinner → versicolor

wider → virginica

🌸 Rule 3 — Large Petals
petal length > 5.15 → virginica


Large petals → always virginica

Correct real-world pattern.

🌳 Why Petal Length Is Root Node

ID3 chooses split with:

maximum information gain


Petal length separates setosa perfectly → highest gain → chosen first.

This confirms your ID3 is working correctly





🔹 Step 1 — Import Libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

✅ What this does

load_iris → loads Iris dataset

DecisionTreeClassifier → builds decision tree

criterion="entropy" later → makes it ID3

plot_tree → draws tree diagram

export_text → prints rules as text

train_test_split → splits data

metrics → check accuracy

🔹 Step 2 — Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

✅ Meaning

X = features (flower measurements)

y = class label (species)

Features:

sepal length
sepal width
petal length
petal width


Classes:

setosa, versicolor, virginica

🔹 Step 3 — Save Names (for readable output)
feature_names = iris.feature_names
class_names = iris.target_names

✅ Why needed

So tree diagram shows:

petal length <= 2.45


instead of:

feature_2 <= 2.45


Makes output understandable.

🔹 Step 4 — Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

✅ Meaning

Split dataset:

75% → training

25% → testing

Why?

👉 Train tree on one part
👉 Test performance on unseen data

random_state → ensures same split every run (important for lab repeatability)

🔹 Step 5 — Build ID3 Tree
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)

✅ This is the core ID3 step
criterion="entropy"

Means:

Use entropy + information gain


That = ID3 algorithm

max_depth=4

Limits tree height.

Why?

Without limit → huge tree

Hard to read

Overfitting risk

Teacher-friendly tree = readable tree.

🔹 Step 6 — Train Model
model.fit(X_train, y_train)

✅ What happens here

Tree learns rules like:

if petal length <= 2.45 → setosa
else if petal width <= 1.75 → versicolor
else → virginica


This is ID3 rule building.

🔹 Step 7 — Prediction
y_pred = model.predict(X_test)

✅ Meaning

Model predicts species for test flowers.

🔹 Step 8 — Accuracy Check
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

✅ Output tells

Accuracy %

Precision

Recall

F1 score

Shows model quality.

🔹 Step 9 — Print Decision Rules (Very Important)
print(export_text(model, feature_names=feature_names))

✅ This prints human-readable rules

Example:

petal length <= 2.45 → setosa
petal width <= 1.75 → versicolor


