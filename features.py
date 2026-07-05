"""
features.py

Feature engineering for the churn model. Goes beyond raw columns to capture
patterns that plausibly relate to churn risk — this is where you show
domain reasoning rather than just one-hot-encoding everything blindly.

Run:
    python src/features.py
"""

import pandas as pd

IN_PATH = "data/enriched_telco.csv"
OUT_PATH = "data/featured_telco.csv"


def add_tenure_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bins = [0, 6, 12, 24, 48, 100]
    labels = ["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4yr+"]
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=False)
    return df


def add_service_bundle_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Customers with more bundled services tend to be stickier.
    Count how many add-on services each customer has.
    """
    df = df.copy()
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["service_count"] = (df[service_cols] == "Yes").sum(axis=1)
    return df


def add_payment_friction_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Electronic check payments correlate with higher churn in this dataset —
    likely a proxy for less "sticky" payment setup (no autopay commitment).
    """
    df = df.copy()
    df["high_friction_payment"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    return df


def add_price_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio of monthly charges to service count — a rough proxy for
    'paying a lot for not much', which may signal price sensitivity.
    """
    df = df.copy()
    df["price_per_service"] = df["MonthlyCharges"] / (df["service_count"] + 1)
    return df


def main():
    df = pd.read_csv(IN_PATH)
    df = add_tenure_buckets(df)
    df = add_service_bundle_count(df)
    df = add_payment_friction_flag(df)
    df = add_price_sensitivity(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"Featured data saved to {OUT_PATH}")
    print(df[["tenure_bucket", "service_count", "high_friction_payment", "price_per_service"]].head())


if __name__ == "__main__":
    main()
