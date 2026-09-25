import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("Data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# First 5 rows
print(df.head())

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove customerID
df = df.drop("customerID", axis=1)

print("\nColumns after removing customerID:")
print(df.columns.tolist())

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

print("\nShape after removing missing TotalCharges:")
print(df.shape)

print("\nMissing TotalCharges after cleaning:")
print(df["TotalCharges"].isnull().sum())

# Check missing values after conversion
print("\nMissing TotalCharges:")
print(df["TotalCharges"].isnull().sum())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns)

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Churn distribution
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nMissing TotalCharges:")
print(df["TotalCharges"].isnull().sum())

# Check data types
print("\nData Types:")
print(df.dtypes)

# Find categorical columns
categorical_columns = df.select_dtypes(include=["object"]).columns

print("\nCategorical Columns:")
print(categorical_columns.tolist())

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Encode target variable
y = y.map({"No": 0, "Yes": 1})

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

print("\nEncoded Features Shape:")
print(X.shape)

print("\nEncoded Feature Columns:")
print(X.columns.tolist())

print("\nEncoded Target:")
print(y.value_counts())

print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nTarget values:")
print(y.value_counts())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining target shape:")
print(y_train.shape)

print("\nTesting target shape:")
print(y_test.shape)

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled Training Data Shape:")
print(X_train_scaled.shape)

print("\nScaled Testing Data Shape:")
print(X_test_scaled.shape)

# Logistic Regression Model
logistic_model = LogisticRegression(max_iter=1000)

# Train the model
logistic_model.fit(X_train_scaled, y_train)

print("\nLogistic Regression model trained successfully!")

# Make predictions
y_pred_logistic = logistic_model.predict(X_test_scaled)

print("\nFirst 10 Predictions:")
print(y_pred_logistic[:10])

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled Training Data Shape:")
print(X_train_scaled.shape)

print("\nScaled Testing Data Shape:")
print(X_test_scaled.shape)


# Logistic Regression Model
logistic_model = LogisticRegression(max_iter=1000)

# Train the model
logistic_model.fit(X_train_scaled, y_train)

print("\nLogistic Regression model trained successfully!")


# Make predictions
y_pred_logistic = logistic_model.predict(X_test_scaled)

print("\nFirst 10 Predictions:")
print(y_pred_logistic[:10])

# Model Evaluation

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Accuracy
accuracy = accuracy_score(y_test, y_pred_logistic)

# Precision
precision = precision_score(y_test, y_pred_logistic)

# Recall
recall = recall_score(y_test, y_pred_logistic)

# F1 Score
f1 = f1_score(y_test, y_pred_logistic)

# ROC-AUC
y_prob_logistic = logistic_model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob_logistic)

print("\n--- Logistic Regression Evaluation ---")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_logistic))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_logistic))


import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred_logistic)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.show()

# ==========================================
# STEP 9 - RANDOM FOREST MODEL
# ==========================================

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Train model
rf_model.fit(X_train_scaled, y_train)

print("\nRandom Forest model trained successfully!")

# Make predictions
y_pred_rf = rf_model.predict(X_test_scaled)

print("\nFirst 10 Random Forest Predictions:")
print(y_pred_rf[:10])

# ==========================================
# STEP 9.2 - RANDOM FOREST EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Calculate evaluation metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Probability predictions for ROC-AUC
y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)

print("\n--- Random Forest Evaluation ---")
print(f"Accuracy  : {accuracy_rf:.4f}")
print(f"Precision : {precision_rf:.4f}")
print(f"Recall    : {recall_rf:.4f}")
print(f"F1 Score  : {f1_rf:.4f}")
print(f"ROC-AUC   : {roc_auc_rf:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# ==========================================
# STEP 9.3 - RANDOM FOREST CONFUSION MATRIX
# ==========================================

cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.show()

# ==========================================
# STEP 10 - MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [accuracy, accuracy_rf],
    "Precision": [precision, precision_rf],
    "Recall": [recall, recall_rf],
    "F1 Score": [f1, f1_rf],
    "ROC-AUC": [roc_auc, roc_auc_rf]
})

print("\n--- Model Comparison ---")
print(comparison.round(4))

# ==========================================
# STEP 11 - FEATURE IMPORTANCE
# ==========================================

# Random Forest Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n--- Top 10 Important Features ---")
print(feature_importance.head(10))

# ==========================================
# STEP 12 - FEATURE IMPORTANCE VISUALIZATION
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.show()

# ==========================================
# STEP 13 - BUSINESS INSIGHTS
# ==========================================

print("\n--- Business Insights ---")

print("1. TotalCharges is the most important feature according to Random Forest.")
print("2. Customer tenure is one of the strongest predictors of churn.")
print("3. MonthlyCharges has a high contribution to the model.")
print("4. Contract type, internet service and payment method also contribute to churn prediction.")
print("5. Customer retention strategies should focus on high-risk customer segments.")

# ==========================================
# STEP 14 - SAVE MODEL AND SCALER
# ==========================================

import joblib

# Save Logistic Regression model
joblib.dump(logistic_model, "logistic_model.pkl")

# Save Random Forest model
joblib.dump(rf_model, "random_forest_model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Save feature columns
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("\n--- Models Saved Successfully ---")
print("Logistic Regression model saved.")
print("Random Forest model saved.")
print("Scaler saved.")
print("Feature columns saved.")