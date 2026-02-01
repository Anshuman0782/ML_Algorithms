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





Output:-
1. If Outlook = Overcast → Play = Yes

2. If Outlook = Sunny AND Humidity = High → Play = No

3. If Outlook = Rain AND Humidity = High AND Wind = Strong → Play = No

4. If Outlook = Rain AND Humidity = High AND Wind = Weak → Play = Yes

5. If Humidity = Normal AND Wind = Weak → Play = Yes

6. If Humidity = Normal AND Wind = Strong AND Temperature = Mild → Play = Yes

7. If Humidity = Normal AND Wind = Strong AND Temperature = Cool/Hot → Play = No


-----------------------------------------------------------------------------------------------------

🌳 What This Tree Represents

This is a Decision Tree built using ID3 (Entropy + Information Gain) to predict:

Play = Yes or No


based on:

Outlook, Temperature, Humidity, Wind


The tree shows:

👉 which feature is checked first
👉 what condition is tested
👉 how data is split
👉 final Yes/No decision

📦 How to Read Each Node Box

Each node shows something like:

Outlook <= 0.5
entropy = 0.94
samples = 14
value = [5, 9]
class = Yes


Let’s decode each line.

🔹 Condition Line (Top Line)

Example:

Outlook <= 0.5


Means:

👉 Tree is splitting on Outlook feature
👉 Because it gave highest information gain (ID3 rule)

Since data was label-encoded:

Example mapping might be:

Overcast = 0
Rain = 1
Sunny = 2


So:

Outlook <= 0.5 → Overcast branch
Outlook > 0.5 → Rain/Sunny branch

🔹 Entropy

Example:

entropy = 0.94


Entropy measures impurity:

Entropy	Meaning
0	Pure (all Yes or all No)
1	Fully mixed

So:

0.94 → mixed Yes/No
0.0 → perfectly pure


Leaves with entropy = 0 are final decisions.

🔹 Samples
samples = 14


Number of rows reaching that node.

Root node = all 14 golf records.

Child nodes = subset after split.

🔹 Value
value = [5, 9]


Counts of each class:

[No, Yes]


So:

5 = No
9 = Yes

🔹 Class
class = Yes


Majority class at that node.

Tree predicts this if it stops here.

🌲 Now Let’s Read Your Tree Logically
🟦 Root Node
Outlook <= 0.5
entropy = 0.94
samples = 14
value = [5,9]
class = Yes


Meaning:

👉 First split chosen = Outlook
👉 Because ID3 found highest information gain here

🟦 Left Branch — Outlook = Overcast
entropy = 0.0
samples = 4
value = [0,4]
class = Yes


✅ All are Yes
✅ Pure node
✅ Leaf node

Rule: If Outlook = Overcast → Play = Yes

🟧 Right Branch — Outlook ≠ Overcast

Next split:

Humidity <= 0.5


Means ID3 next best feature = Humidity.

🟧 Humidity High Branch

Leads mostly to:

class = No


Rule:

If Outlook = Sunny AND Humidity = High → No

🟦 Humidity Normal Branch

Next split:

Wind <= 0.5


Means wind decides here.

🟦 Wind Weak
class = Yes

🟧 Wind Strong
class = No


Rule:

If Humidity = Normal AND Wind = Weak → Yes
If Humidity = Normal AND Wind = Strong → No

📝 Final Decision Rules (From Your Tree)

You can write this in exam:

1. If Outlook = Overcast → Play = Yes
2. If Outlook = Sunny AND Humidity = High → Play = No
3. If Outlook = Sunny AND Humidity = Normal → Play = Yes
4. If Outlook = Rain AND Wind = Strong → Play = No
5. If Outlook = Rain AND Wind = Weak → Play = Yes
