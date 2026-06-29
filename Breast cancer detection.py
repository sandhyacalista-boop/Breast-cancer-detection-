# ==========================================================
# Breast Cancer Detection using Machine Learning
# Using data.csv
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("data.csv")

print("First 5 Rows:\n")
print(df.head())

# ==========================================================
# Remove Unnecessary Columns
# ==========================================================

df.drop(columns=["id", "Unnamed: 32"], inplace=True)

# ==========================================================
# Convert Diagnosis to Numbers
# M = 1 (Malignant)
# B = 0 (Benign)
# ==========================================================

encoder = LabelEncoder()
df["diagnosis"] = encoder.fit_transform(df["diagnosis"])

print("\nDataset Information:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nStatistical Summary:\n")
print(df.describe())

# ==========================================================
# Visualize Target Column
# ==========================================================

plt.figure(figsize=(6,4))
sns.countplot(x="diagnosis", data=df)
plt.title("Diagnosis Distribution")
plt.xlabel("0 = Benign | 1 = Malignant")
plt.ylabel("Count")
plt.show()

# ==========================================================
# Split Features and Target
# ==========================================================

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# Feature Scaling
# ==========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================================
# Train Model
# ==========================================================

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

# ==========================================================
# Prediction
# ==========================================================

y_pred = model.predict(X_test)

# ==========================================================
# Accuracy
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("==============================")

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================================================
# Classification Report
# ==========================================================

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================================================
# Predict One Sample
# ==========================================================

sample = X.iloc[0].values.reshape(1, -1)
sample = scaler.transform(sample)

prediction = model.predict(sample)

print("\nPrediction for First Patient:")

if prediction[0] == 1:
    print("Malignant (Cancer Detected)")
else:
    print("Benign (No Cancer Detected)")

# ==========================================================
# Test with Custom Input (Optional)
# ==========================================================

print("\nProject Completed Successfully!")