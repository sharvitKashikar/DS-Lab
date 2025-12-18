# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# ================================
# 2. LOAD DATA
# ================================
# Sample dataset created manually (can be replaced with CSV)
data = {
    "Hours_Studied": [2, 4, 6, 8, 10, 3, 7, 5, 9, 1],
    "Attendance": [60, 70, 75, 85, 90, 65, 80, 72, 88, 55],
    "Previous_Score": [45, 50, 60, 70, 78, 48, 65, 58, 75, 40],
    "Internet_Access": ["Yes", "Yes", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "No"],
    "Final_Score": [50, 55, 65, 75, 85, 52, 72, 60, 80, 45]
}

df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())

# ================================
# 3. DATA CLEANING
# ================================
print("\nMissing Values:")
print(df.isnull().sum())

# Encode categorical data
encoder = LabelEncoder()
df["Internet_Access"] = encoder.fit_transform(df["Internet_Access"])

# ================================
# 4. EXPLORATORY DATA ANALYSIS
# ================================
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

sns.pairplot(df)
plt.show()

# ================================
# 5. FEATURE SELECTION
# ================================
X = df.drop("Final_Score", axis=1)
y = df["Final_Score"]

# ================================
# 6. FEATURE SCALING
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================
# 7. TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ================================
# 8. MODEL 1: LINEAR REGRESSION
# ================================
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

# ================================
# 9. MODEL 2: RANDOM FOREST
# ================================
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

# ================================
# 10. MODEL EVALUATION
# ================================
def evaluate_model(name, y_test, y_pred):
    print(f"\n{name} Performance:")
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2 Score:", r2_score(y_test, y_pred))

evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)

# ================================
# 11. FEATURE IMPORTANCE
# ================================
importance = rf.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importance, y=features)
plt.title("Feature Importance (Random Forest)")
plt.show()

# ================================
# 12. PREDICTION ON NEW DATA
# ================================
new_student = pd.DataFrame({
    "Hours_Studied": [6],
    "Attendance": [78],
    "Previous_Score": [65],
    "Internet_Access": [1]
})

new_student_scaled = scaler.transform(new_student)
predicted_score = rf.predict(new_student_scaled)

print("\nPredicted Final Score:", round(predicted_score[0], 2))
