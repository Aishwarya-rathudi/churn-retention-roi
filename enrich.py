"""
enrich.py

Takes the raw Telco churn dataset and adds two things the original data doesn't have:
  1. Customer Lifetime Value (CLV) — a proxy based on tenure and monthly spend
  2. Retention action costs — what it costs to try to save this customer

This is the step that turns a plain classification dataset into a business
decision-making problem. Real companies would pull CLV from finance systems
and cost estimates from marketing/ops — here we simulate reasonable proxies
and explain the assumptions so it's defensible in an interview.

Run:
    python src/enrich.py
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/raw_telco.csv"
OUT_PATH = "data/enriched_telco.csv"

# --- Assumptions (state these explicitly in your write-up) ---
DISCOUNT_PCT = 0.20        # retention discount: 20% off monthly charges
DISCOUNT_MONTHS = 3        # discount applied for 3 months
OUTREACH_COST = 15.0       # flat cost of a retention phone call / email campaign
PROJECTED_LIFETIME_MONTHS = 24  # used to estimate forward-looking CLV


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Telco dataset quirk: TotalCharges is sometimes a blank string, not NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    return df


def add_clv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple CLV proxy: historical spend so far + projected future spend
    based on monthly charges and an assumed forward-looking horizon,
    discounted slightly for customers already showing churn risk signals
    (month-to-month contract, low tenure).
    """
    df = df.copy()
    df["historical_value"] = df["TotalCharges"]
    df["projected_future_value"] = df["MonthlyCharges"] * PROJECTED_LIFETIME_MONTHS

    # simple risk discount: month-to-month contracts are less "sticky"
    risk_discount = np.where(df["Contract"] == "Month-to-month", 0.6, 1.0)
    df["projected_future_value"] *= risk_discount

    df["CLV"] = df["historical_value"] + df["projected_future_value"]
    return df


def add_retention_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two retention actions are simulated:
      - 'discount': temporary price cut, costs a % of monthly revenue over N months
      - 'outreach': a flat-cost personal touch (call/email), cheaper but less powerful
    In train.py / simulate.py you'll choose the better action per customer.
    """
    df = df.copy()
    df["cost_discount"] = df["MonthlyCharges"] * DISCOUNT_PCT * DISCOUNT_MONTHS
    df["cost_outreach"] = OUTREACH_COST
    return df


def main():
    df = load_raw(RAW_PATH)
    df = add_clv(df)
    df = add_retention_costs(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"Enriched data saved to {OUT_PATH}")
    print(df[["CLV", "cost_discount", "cost_outreach"]].describe())


if __name__ == "__main__":
    main()
