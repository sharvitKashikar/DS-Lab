# --------------------------------------
# Agriculture Data Science Project
# Crop Yield Prediction using ML
# --------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------
# Step 1: Create Sample Dataset
# -------------------------------

np.random.seed(42)

data = {
    "Rainfall_mm": np.random.randint(400, 2000, 150),
    "Temperature_C": np.random.uniform(15, 40, 150),
    "Soil_Nitrogen": np.random.uniform(20, 80, 150),
    "Soil_Phosphorus": np.random.uniform(10, 50, 150),
    "Soil_Potassium": np.random.uniform(10, 60, 150),
    "Crop_Yield_ton_per_hectare": np.random.uniform(1.5, 6.0, 150)
}

df = pd.DataFrame(data)

# -------------------------------
# Step 2: Exploratory Data Analysis
# -------------------------------

print("Dataset Preview:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe())

# Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# -------------------------------
# Step 3: Feature & Target Split
# -------------------------------

X = df.drop("Crop_Yield_ton_per_hectare", axis=1)
y = df["Crop_Yield_ton_per_hectare"]

# -------------------------------
# Step 4: Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 5: Model Training
# -------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# Step 6: Model Evaluation
# -------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# -------------------------------
# Step 7: Feature Importance
# -------------------------------

importance = model.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importance)
plt.title("Feature Importance for Crop Yield")
plt.xlabel("Importance Score")
plt.show()

# -------------------------------
# Step 8: Predict for New Input
# -------------------------------

new_farm_data = pd.DataFrame({
    "Rainfall_mm": [1200],
    "Temperature_C": [28],
    "Soil_Nitrogen": [60],
    "Soil_Phosphorus": [35],
    "Soil_Potassium": [40]
})

predicted_yield = model.predict(new_farm_data)

print("\nPredicted Crop Yield:")
print(f"{predicted_yield[0]:.2f} tons per hectare")
