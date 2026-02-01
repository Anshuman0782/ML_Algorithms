import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# -----------------------------
# Step 1: Read CSV File
# -----------------------------
df = pd.read_csv("buy_computer.csv")

# -----------------------------
# Step 2: Encode Categorical Data
# -----------------------------
label_encoders = {}

for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# -----------------------------
# Step 3: Features and Target
# -----------------------------
X = df[['Age', 'Income', 'Student', 'Credit_Rating']]
y = df['Buy_Computer']

# -----------------------------
# Step 4: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 5: Train Naïve Bayes Model
# -----------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# -----------------------------
# Step 6: Predict New Data
# -----------------------------
unknown_samples = pd.DataFrame({
    'Age': ['<=30', '>40'],
    'Income': ['Medium', 'Low'],
    'Student': ['Yes', 'No'],
    'Credit_Rating': ['Fair', 'Excellent']
})

# Encode unknown samples
for column in unknown_samples.columns:
    unknown_samples[column] = label_encoders[column].transform(unknown_samples[column])

# -----------------------------
# Step 7: SHOW CALCULATION (Probabilities)
# -----------------------------
probabilities = model.predict_proba(unknown_samples)

prob_df = pd.DataFrame(
    probabilities,
    columns=label_encoders['Buy_Computer'].classes_
)

print("\nPosterior Probability Calculation:")
print(prob_df)

# -----------------------------
# Step 8: FINAL PREDICTION
# -----------------------------
predictions = model.predict(unknown_samples)
final_result = label_encoders['Buy_Computer'].inverse_transform(predictions)

print("\nFinal Prediction:")
print(final_result)
