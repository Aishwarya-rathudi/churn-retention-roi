"""
train.py

Trains and compares two models: logistic regression (interpretable baseline)
and XGBoost (stronger performer). Evaluates on PR-AUC, not accuracy —
churn is imbalanced, so accuracy is a misleading metric here. Explaining
this choice in your write-up is a strong signal of statistical maturity.

Run:
    python src/train.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, classification_report
)
from xgboost import XGBClassifier

IN_PATH = "data/featured_telco.csv"
MODEL_OUT = "data/model.joblib"

TARGET = "Churn"

NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "service_count", "price_per_service", "high_friction_payment",
]
CATEGORICAL_FEATURES = [
    "Contract", "InternetService", "PaymentMethod", "tenure_bucket",
]


def load_data():
    df = pd.read_csv(IN_PATH)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    # keep CLV/cost columns aside for the simulation step later
    aux = df[["customerID", "CLV", "cost_discount", "cost_outreach"]] if "customerID" in df.columns else df[["CLV", "cost_discount", "cost_outreach"]]
    return X, y, aux, df


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def main():
    X, y, aux, df = load_data()
    X_train, X_test, y_train, y_test, aux_train, aux_test = train_test_split(
        X, y, aux, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    # --- Baseline: Logistic Regression ---
    logreg = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    logreg.fit(X_train, y_train)
    logreg_probs = logreg.predict_proba(X_test)[:, 1]

    # --- Stronger model: XGBoost ---
    xgb = Pipeline([
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric="logloss", random_state=42,
        )),
    ])
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]

    print("=== Logistic Regression ===")
    print(f"ROC-AUC: {roc_auc_score(y_test, logreg_probs):.3f}")
    print(f"PR-AUC:  {average_precision_score(y_test, logreg_probs):.3f}")

    print("\n=== XGBoost ===")
    print(f"ROC-AUC: {roc_auc_score(y_test, xgb_probs):.3f}")
    print(f"PR-AUC:  {average_precision_score(y_test, xgb_probs):.3f}")
    print("\nClassification report (XGBoost, threshold=0.5):")
    print(classification_report(y_test, (xgb_probs > 0.5).astype(int)))

    # Save the better model (XGBoost, typically) for use in simulate.py / app
    joblib.dump(xgb, MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")

    # Save test set predictions + aux data for simulate.py
    results = aux_test.copy()
    results["churn_prob"] = xgb_probs
    results["actual_churn"] = y_test.values
    results.to_csv("data/test_predictions.csv", index=False)
    print("Test predictions saved to data/test_predictions.csv")


if __name__ == "__main__":
    main()
