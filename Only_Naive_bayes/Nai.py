import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Step 1: Load Boston from OpenML
# -----------------------------
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame

print("Dataset shape:", df.shape)

# -----------------------------
# Step 2: Convert MEDV to Classes
# -----------------------------
# create price category for classification
df["Price_Class"] = pd.qcut(
    df["MEDV"],
    q=3,
    labels=["Low", "Medium", "High"]
)

# -----------------------------
# Step 3: Features and Target
# -----------------------------
X = df.drop(columns=["MEDV", "Price_Class"])
y = df["Price_Class"]

# -----------------------------
# Step 4: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# Step 5: Standardize Features
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Step 6: Train Naive Bayes
# -----------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# -----------------------------
# Step 7: Posterior Probabilities
# -----------------------------
probs = model.predict_proba(X_test[:5])

prob_df = pd.DataFrame(
    probs,
    columns=model.classes_
)

print("\nPosterior Probabilities (first 5 test samples):")
print(prob_df)

# -----------------------------
# Step 8: Final Prediction
# -----------------------------
pred = model.predict(X_test[:5])

print("\nPredicted Classes:")
print(pred)
