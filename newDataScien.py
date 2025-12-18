"""
Modern Data Science Pipeline (2025-ready)
- Polars for fast data processing
- LightGBM for high-performance ML
- Optuna for hyperparameter optimization
- SHAP for explainable AI
"""

# ===============================
# 1. IMPORTS
# ===============================
import polars as pl
import numpy as np
import lightgbm as lgb
import optuna
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler


# ===============================
# 2. DATA GENERATION (SIMULATED REALISTIC DATA)
# ===============================
def generate_customer_data(n=10000):
    rng = np.random.default_rng(42)

    data = {
        "age": rng.integers(18, 70, n),
        "monthly_charges": rng.uniform(20, 120, n),
        "tenure_months": rng.integers(1, 72, n),
        "support_tickets": rng.poisson(2, n),
        "contract_type": rng.choice(["monthly", "yearly"], n, p=[0.7, 0.3]),
        "internet_service": rng.choice(["dsl", "fiber", "none"], n),
    }

    churn_prob = (
        0.4
        - 0.004 * data["tenure_months"]
        + 0.002 * data["monthly_charges"]
        + 0.08 * data["support_tickets"]
    )

    churn = (rng.random(n) < churn_prob).astype(int)
    data["churn"] = churn

    return pl.DataFrame(data)


df = generate_customer_data()

print("Sample Data:")
print(df.head())


# ===============================
# 3. FEATURE ENGINEERING
# ===============================
df = df.with_columns([
    (pl.col("monthly_charges") * pl.col("tenure_months")).alias("lifetime_value"),
    (pl.col("support_tickets") / (pl.col("tenure_months") + 1)).alias("tickets_per_month"),
])

df = df.to_dummies(columns=["contract_type", "internet_service"])

print("\nAfter Feature Engineering:")
print(df.head())


# ===============================
# 4. TRAIN / TEST SPLIT
# ===============================
target = "churn"
features = [c for c in df.columns if c != target]

X = df.select(features).to_numpy()
y = df.select(target).to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ===============================
# 5. OPTUNA HYPERPARAMETER TUNING
# ===============================
def objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "verbosity": -1,
    }

    model = lgb.LGBMClassifier(**params, n_estimators=300)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]

    return roc_auc_score(y_test, preds)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("\nBest Hyperparameters:")
print(study.best_params)


# ===============================
# 6. FINAL MODEL TRAINING
# ===============================
final_model = lgb.LGBMClassifier(
    **study.best_params,
    n_estimators=500
)

final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]

print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))


# ===============================
# 7. EXPLAINABLE AI WITH SHAP
# ===============================
explainer = shap.Explainer(final_model, X_train)
shap_values = explainer(X_test[:300])

shap.summary_plot(
    shap_values,
    X_test[:300],
    feature_names=features,
    show=False
)

plt.tight_layout()
plt.show()


# ===============================
# 8. PRODUCTION-STYLE PREDICTION FUNCTION
# ===============================
def predict_churn(customer_dict):
    customer_df = pl.DataFrame([customer_dict])
    customer_df = customer_df.with_columns([
        (pl.col("monthly_charges") * pl.col("tenure_months")).alias("lifetime_value"),
        (pl.col("support_tickets") / (pl.col("tenure_months") + 1)).alias("tickets_per_month"),
    ])

    customer_df = customer_df.to_dummies()

    # Align missing columns
    for col in features:
        if col not in customer_df.columns:
            customer_df = customer_df.with_columns(pl.lit(0).alias(col))

    customer_df = customer_df.select(features)
    customer_scaled = scaler.transform(customer_df.to_numpy())

    prob = final_model.predict_proba(customer_scaled)[0][1]
    return prob


# ===============================
# 9. SAMPLE PREDICTION
# ===============================
sample_customer = {
    "age": 29,
    "monthly_charges": 95,
    "tenure_months": 5,
    "support_tickets": 4,
    "contract_type": "monthly",
    "internet_service": "fiber"
}

print("\nChurn Probability:", predict_churn(sample_customer))
