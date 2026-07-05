"""
simulate.py

THE KEY DIFFERENTIATOR of this project.

Most churn projects stop after train.py: "here's who's likely to churn."
This script answers the actual business question: "who is worth
intervening on, and with which action, GIVEN A LIMITED BUDGET?"

For each customer, we calculate the expected value of each retention action:

    expected_value(action) = P(churn) * CLV_saved_if_retained - cost(action)

We only intervene where expected_value > 0, and we pick whichever action
(discount vs outreach) has the higher expected value for that customer.
This is a simplified expected-value framework — in a real company you'd
also model the *effectiveness* of each action (does a discount actually
reduce churn probability, and by how much?), which is a great "next steps"
talking point in your write-up.

IMPORTANT: without a budget constraint, "value-based ranking" and "churn
probability ranking" converge whenever nearly everyone is worth targeting
(expected_value > 0 for almost all customers). The real-world case for
value-based targeting only shows up when you can't target everyone — e.g.
a retention team that can only run outreach on a few hundred customers a
month. BUDGET_SIZE below simulates that constraint so the two strategies
actually diverge and the comparison means something.

Run:
    python src/simulate.py
"""

import pandas as pd

IN_PATH = "data/test_predictions.csv"
OUT_PATH = "data/intervention_plan.csv"

# Assumed effectiveness of each action: how much it reduces churn probability.
# In reality you'd estimate this from historical A/B tests. Here it's a
# reasonable assumption — state it clearly as such in your write-up.
DISCOUNT_EFFECTIVENESS = 0.35   # cuts churn probability by 35% (relative)
OUTREACH_EFFECTIVENESS = 0.15   # cuts churn probability by 15% (relative)

# How many customers the retention team can realistically afford to target
# this cycle (personalized outreach doesn't scale infinitely). Set to None
# to disable the constraint and target everyone with positive EV.
BUDGET_SIZE = 200


def compute_expected_value(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Value at stake if this customer churns and we do nothing
    df["value_at_risk"] = df["churn_prob"] * df["CLV"]

    # Expected value of each action = (reduction in churn prob) * CLV - cost
    df["ev_discount"] = (
        df["churn_prob"] * DISCOUNT_EFFECTIVENESS * df["CLV"] - df["cost_discount"]
    )
    df["ev_outreach"] = (
        df["churn_prob"] * OUTREACH_EFFECTIVENESS * df["CLV"] - df["cost_outreach"]
    )

    # Best action = whichever has higher expected value (must be positive to act)
    def pick_action(row):
        options = {"discount": row["ev_discount"], "outreach": row["ev_outreach"]}
        best_action = max(options, key=options.get)
        best_value = options[best_action]
        if best_value <= 0:
            return pd.Series(["no_action", 0.0])
        return pd.Series([best_action, best_value])

    df[["recommended_action", "expected_value"]] = df.apply(pick_action, axis=1)
    return df


def main():
    df = pd.read_csv(IN_PATH)
    df = compute_expected_value(df)
    df = df.sort_values("expected_value", ascending=False)
    df.to_csv(OUT_PATH, index=False)

    n_positive_ev = (df["recommended_action"] != "no_action").sum()

    print(f"Customers evaluated: {len(df)}")
    print(f"Customers with positive expected value (uncapped): {n_positive_ev}")
    print("\nTop 10 highest-value customers to target:")
    print(df[["churn_prob", "CLV", "recommended_action", "expected_value"]].head(10))

    # --- Budget-constrained comparison ---
    # This is the part that actually demonstrates the value of ranking by
    # expected value rather than churn probability alone. Without a budget
    # cap, both approaches tend to target almost everyone with positive EV,
    # so they converge and "improvement" looks artificially small.
    if BUDGET_SIZE is not None:
        n = min(BUDGET_SIZE, len(df))

        smart_targets = df.sort_values("expected_value", ascending=False).head(n)
        smart_ev = smart_targets["expected_value"].sum()

        naive_targets = df.sort_values("churn_prob", ascending=False).head(n)
        naive_ev = naive_targets["expected_value"].sum()

        improvement = smart_ev - naive_ev
        pct_improvement = (improvement / naive_ev * 100) if naive_ev != 0 else float("nan")

        print(f"\n--- Budget-constrained comparison (top {n} customers only) ---")
        print(f"Value-based targeting expected value: ${smart_ev:,.2f}")
        print(f"Naive (churn-probability-only) targeting expected value: ${naive_ev:,.2f}")
        print(f"Improvement: ${improvement:,.2f} ({pct_improvement:.1f}% better)")

        overlap = len(set(smart_targets.index) & set(naive_targets.index))
        print(f"Overlap between the two target lists: {overlap}/{n} customers "
              f"({overlap / n:.0%}) — the rest is where the two strategies disagree.")
    else:
        total_ev = df["expected_value"].sum()
        print(f"\nTotal expected value from all positive-EV interventions: ${total_ev:,.2f}")
        print("(BUDGET_SIZE is None — no budget constraint applied.)")


if __name__ == "__main__":
    main()