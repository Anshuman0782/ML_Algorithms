import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Step 1: Golf Playing Dataset
# -----------------------------
data = {
    "Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast",
                "Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"],

    "Temperature": ["Hot","Hot","Hot","Mild","Cool","Cool","Cool",
                    "Mild","Cool","Mild","Mild","Mild","Hot","Mild"],

    "Humidity": ["High","High","High","High","Normal","Normal","Normal",
                 "High","Normal","Normal","Normal","High","Normal","High"],

    "Wind": ["Weak","Strong","Weak","Weak","Weak","Strong","Strong",
             "Weak","Weak","Weak","Strong","Strong","Weak","Strong"],

    "Play": ["No","No","Yes","Yes","Yes","No","Yes",
             "No","Yes","Yes","Yes","Yes","Yes","No"]
}

df = pd.DataFrame(data)

print("Dataset:\n", df)

# -----------------------------
# Step 2: Encode Categorical Data
# -----------------------------
df_enc = df.copy()
encoders = {}

for col in df.columns:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df[col])
    encoders[col] = le

# -----------------------------
# Step 3: Features and Target
# -----------------------------
X = df_enc.drop("Play", axis=1)
y = df_enc["Play"]

# -----------------------------
# Step 4: ID3 = Entropy Tree
# -----------------------------
model = DecisionTreeClassifier(
    criterion="entropy",   # ID3
    max_depth=4,
    random_state=0
)

model.fit(X, y)

# -----------------------------
# Step 5: Print Decision Rules
# -----------------------------
print("\nDecision Rules:\n")
print(export_text(model, feature_names=list(X.columns)))

# -----------------------------
# Step 6: Draw Decision Tree
# -----------------------------
plt.figure(figsize=(14, 9))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=encoders["Play"].classes_,
    filled=True,
    rounded=True
)

plt.title("ID3 Decision Tree — Golf Playing Dataset")
plt.show()
