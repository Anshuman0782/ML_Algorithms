import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Buy Computer Dataset
# -----------------------------
data = {
    'Age': ['<=30', '<=30', '31-40', '>40', '>40', '>40', '31-40',
            '<=30', '<=30', '>40', '<=30', '31-40', '31-40', '>40'],

    'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low',
               'Medium', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'Medium'],

    'Student': ['No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes',
                'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No'],

    'Credit_Rating': ['Fair', 'Excellent', 'Fair', 'Fair', 'Fair', 'Excellent',
                      'Excellent', 'Fair', 'Fair', 'Fair', 'Excellent',
                      'Excellent', 'Fair', 'Excellent'],

    'Buy_Computer': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
                     'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# -----------------------------
# Encode each column separately
# -----------------------------
df_encoded = df.copy()
encoders = {}

for col in df.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    encoders[col] = le   # store encoder if needed later

# -----------------------------
# Features and Target
# -----------------------------
X = df_encoded.drop('Buy_Computer', axis=1)
y = df_encoded['Buy_Computer']

# -----------------------------
# CART Model (Gini Index)
# -----------------------------
model = DecisionTreeClassifier(
    criterion='gini',     # CART uses Gini
    max_depth=4,          # keeps tree readable
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

plt.title("CART Decision Tree — Buy Computer Dataset")
plt.show()

# -----------------------------
# Print Decision Rules
# -----------------------------
print("\nDecision Tree Rules:\n")
rules = export_text(model, feature_names=list(X.columns))
print(rules)
