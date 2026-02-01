import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load dataset from CSV
# -----------------------------
df = pd.read_csv("buy_computer.csv")

print("Dataset Preview:\n")
print(df.head())

# -----------------------------
# Encode each column separately
# -----------------------------
df_encoded = df.copy()
encoders = {}

for col in df.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    encoders[col] = le

# -----------------------------
# Features and Target
# -----------------------------
X = df_encoded.drop('Buy_Computer', axis=1)
y = df_encoded['Buy_Computer']

# -----------------------------
# CART Model (Gini Index)
# -----------------------------
model = DecisionTreeClassifier(
    criterion='gini',   # CART
    max_depth=4,
    random_state=42
)

model.fit(X, y)

# -----------------------------
# Plot Decision Tree
# -----------------------------
plt.figure(figsize=(14, 9))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=encoders['Buy_Computer'].classes_,
    filled=True,
    rounded=True
)

plt.title("CART Decision Tree — Buy Computer Dataset (From CSV)")
plt.show()

# -----------------------------
# Print Decision Rules
# -----------------------------
print("\nDecision Tree Rules:\n")
print(export_text(model, feature_names=list(X.columns)))
