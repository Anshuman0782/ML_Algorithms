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