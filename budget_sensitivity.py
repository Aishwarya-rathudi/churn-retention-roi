"""
budget_sensitivity.py

Runs the value-based vs. naive targeting comparison across a range of
budget sizes, and plots how the improvement changes. This answers the
natural follow-up question to simulate.py: "does value-based targeting
help at every budget size, or just this one?"

Expected pattern: the improvement is largest at small/moderate budgets
(where you're forced to be selective) and shrinks toward zero as the
budget approaches "target everyone" — because at that point there's no
real choice left to make. This U-shaped intuition (constrained → high
value, unconstrained → low value) is a great thing to show and explain
in an interview.

Run:
    python src/budget_sensitivity.py
"""

import pandas as pd
import matplotlib.pyplot as plt

from simulate import compute_expected_value, IN_PATH  # reuse existing logic

BUDGET_SIZES = [50, 100, 200, 300, 500, 750, 1000, 1271]  # 1271 = all positive-EV customers
OUT_CHART = "data/budget_sensitivity.png"
OUT_CSV = "data/budget_sensitivity.csv"


def run_sensitivity(df: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    results = []
    for n in budgets:
        n = min(n, len(df))

        smart_targets = df.sort_values("expected_value", ascending=False).head(n)
        smart_ev = smart_targets["expected_value"].sum()

        naive_targets = df.sort_values("churn_prob", ascending=False).head(n)
        naive_ev = naive_targets["expected_value"].sum()

        improvement = smart_ev - naive_ev
        pct_improvement = (improvement / naive_ev * 100) if naive_ev != 0 else float("nan")
        overlap = len(set(smart_targets.index) & set(naive_targets.index))

        results.append({
            "budget_size": n,
            "smart_ev": smart_ev,
            "naive_ev": naive_ev,
            "improvement": improvement,
            "pct_improvement": pct_improvement,
            "overlap_pct": overlap / n * 100,
        })

    return pd.DataFrame(results)


def plot_results(results: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(results["budget_size"], results["smart_ev"], marker="o", label="Value-based targeting")
    ax1.plot(results["budget_size"], results["naive_ev"], marker="o", label="Naive (churn prob only)")
    ax1.set_xlabel("Budget size (number of customers targeted)")
    ax1.set_ylabel("Total expected value ($)")
    ax1.set_title("Expected Value by Targeting Strategy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(results["budget_size"], results["pct_improvement"], marker="o", color="darkgreen")
    ax2.set_xlabel("Budget size (number of customers targeted)")
    ax2.set_ylabel("Improvement over naive (%)")
    ax2.set_title("Value-Based Targeting Advantage Shrinks as Budget Grows")
    ax2.grid(alpha=0.3)
    ax2.axhline(0, color="gray", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(OUT_CHART, dpi=150)
    print(f"Chart saved to {OUT_CHART}")


def main():
    df = pd.read_csv(IN_PATH)
    df = compute_expected_value(df)

    results = run_sensitivity(df, BUDGET_SIZES)
    results.to_csv(OUT_CSV, index=False)

    print(results.to_string(index=False))
    print(f"\nResults saved to {OUT_CSV}")

    plot_results(results)


if __name__ == "__main__":
    main()