"""
explain.py

Adds model explainability using SHAP (SHapley Additive exPlanations).

Two outputs:
  1. GLOBAL feature importance — which features matter most across all
     customers (a summary plot). Answers: "what drives churn in general?"
  2. PER-CUSTOMER explanations — for individual customers, which specific
     features pushed their churn probability up or down, and by how much.
     Answers: "why did the model flag THIS customer?"

This is the difference between a model that outputs a number and a model
you can actually explain to a stakeholder (or defend in an interview).

Run:
    python src/explain.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

MODEL_PATH = "data/model.joblib"
DATA_PATH = "data/featured_telco.csv"

GLOBAL_CHART_OUT = "data/shap_global_importance.png"
WATERFALL_CHART_OUT = "data/shap_customer_example.png"

NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "service_count", "price_per_service", "high_friction_payment",
]
CATEGORICAL_FEATURES = [
    "Contract", "InternetService", "PaymentMethod", "tenure_bucket",
]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_model_and_data():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS]
    return model, X, df


def get_transformed_features(pipeline, X):
    """
    The trained model is a Pipeline: [preprocessor, classifier].
    SHAP's TreeExplainer needs the classifier alone, and the data
    already transformed (scaled numerics + one-hot encoded categoricals).
    This function does that transformation and recovers readable
    feature names for the one-hot encoded columns.
    """
    preprocessor = pipeline.named_steps["prep"]
    classifier = pipeline.named_steps["clf"]

    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    # Recover feature names after ColumnTransformer (numeric passthrough + one-hot)
    num_names = NUMERIC_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = num_names + cat_names

    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
    return classifier, X_transformed_df


def plot_global_importance(explainer, X_transformed_df, shap_values):
    plt.figure()
    shap.summary_plot(shap_values, X_transformed_df, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(GLOBAL_CHART_OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Global feature importance chart saved to {GLOBAL_CHART_OUT}")


def plot_customer_example(base_values, X_transformed_df, shap_values, df, customer_idx=None):
    """
    Picks a high-churn-probability, high-CLV customer by default (the kind
    a retention team would actually care about), and shows a waterfall plot
    of exactly which features pushed their prediction up or down.

    base_values: per-sample expected/base value array from the Explanation
    object (shap.Explainer gives one base value per sample, unlike
    TreeExplainer's single scalar — using per-sample values here is correct
    for both model types).
    """
    if customer_idx is None:
        # Pick a customer worth explaining: high churn risk, high CLV
        candidate = df.copy()
        candidate["_score"] = candidate.get("CLV", 0) * candidate.get("churn_prob", 0.5)
        customer_idx = candidate["_score"].idxmax()

    # customer_idx is a dataframe index label; map it to a positional index
    # for numpy array indexing into shap_values/base_values.
    position = df.index.get_loc(customer_idx)

    base_value = base_values[position]
    base_value = base_value[0] if hasattr(base_value, "__len__") else base_value

    plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        base_value,
        shap_values[position],
        feature_names=X_transformed_df.columns.tolist(),
        max_display=12,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(WATERFALL_CHART_OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-customer explanation chart saved to {WATERFALL_CHART_OUT} "
          f"(customer index {customer_idx})")


def main():
    model, X, df = load_model_and_data()
    classifier, X_transformed_df = get_transformed_features(model, X)

    # shap.Explainer (not TreeExplainer) auto-dispatches to the right
    # algorithm for whichever model type was selected by train.py's
    # comparison — TreeExplainer alone would silently break if Logistic
    # Regression or another non-tree model won the comparison.
    background = X_transformed_df.sample(min(100, len(X_transformed_df)), random_state=42)
    explainer = shap.Explainer(classifier, background)
    explanation = explainer(X_transformed_df)
    shap_values = explanation.values
    base_values = explanation.base_values

    # Some explainer/model combinations return a 3D array (n_samples,
    # n_features, n_classes) for binary classifiers instead of 2D — take
    # the positive class (index 1) if so, for consistency downstream.
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    if hasattr(base_values, "ndim") and base_values.ndim == 2:
        base_values = base_values[:, 1]

    # Global importance: mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_transformed_df.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print("Top 10 most important features (by mean |SHAP value|):")
    print(importance_df.head(10).to_string(index=False))

    plot_global_importance(explainer, X_transformed_df, shap_values)

    # Add churn_prob to df for picking an interesting example customer
    df = df.copy()
    df["churn_prob"] = model.predict_proba(X)[:, 1]
    plot_customer_example(base_values, X_transformed_df, shap_values, df)

    importance_df.to_csv("data/feature_importance.csv", index=False)
    print("\nFeature importance table saved to data/feature_importance.csv")


if __name__ == "__main__":
    main()