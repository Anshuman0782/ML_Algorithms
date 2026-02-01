import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -----------------------------
# Create DataFrame
# -----------------------------
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [35, 40, 50, 55, 65, 70, 75, 85, 90, 95]
}

df = pd.DataFrame(data)

print("Dataset:\n", df)

# -----------------------------
# Features and Target
# -----------------------------
X = df[["Hours_Studied"]]   # must be 2D
y = df["Marks"]

# -----------------------------
# Train Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X, y)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X)

# -----------------------------
# Model Parameters
# -----------------------------
print("\nSlope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
print("R2 Score:", r2_score(y, y_pred))

# -----------------------------
# Plot Scatter + Regression Line
# -----------------------------
plt.figure(figsize=(8, 5))

# actual points
plt.scatter(X, y, label="Actual Data")

# regression line
plt.plot(X, y_pred, label="Regression Line")

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression — Study Hours vs Marks")
plt.legend()
plt.show()