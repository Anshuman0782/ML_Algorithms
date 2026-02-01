import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("diabetes.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop("Outcome", axis=1)   # features
y = df["Outcome"]                # class label (Diabetes)

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)

# -----------------------------
# CART Model (Gini Index)
# -----------------------------
model = DecisionTreeClassifier(
    criterion="gini",   # CART
    max_depth=4,        # keeps tree readable for drawing
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Predictions & Accuracy
# -----------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Plot Decision Tree
# -----------------------------
plt.figure(figsize=(16, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Diabetes", "Diabetes"],
    filled=True,
    rounded=True
)

plt.title("CART Decision Tree — Pima Indians Diabetes Dataset")
plt.show()

# -----------------------------
# Print Decision Rules
# -----------------------------
print("\nDecision Tree Rules:\n")
print(export_text(model, feature_names=list(X.columns)))
