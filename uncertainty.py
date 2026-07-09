"""
uncertainty.py

Adds confidence intervals to revenue-saved estimates via Monte Carlo
simulation, instead of reporting a single point estimate as if it were
certain.

WHY THIS MATTERS: "expected_value = churn_prob * effectiveness * CLV - cost"
is an EXPECTATION, not a guaranteed outcome. Two separate sources of
randomness are baked into it:
  1. Did this customer actually churn or not (a Bernoulli draw at rate
     churn_prob)?
  2. If targeted, did the retention action actually work (a Bernoulli draw
     at rate `effectiveness`)?

Reporting only the mean of these outcomes hides how much they could
plausibly vary. This module runs the random draws many times (Monte Carlo
simulation) and reports the mean alongside a 95% interval — the range
within which the realized outcome would fall 95% of the time, given the
model's assumptions. This is standard practice when presenting any
forecast that depends on probabilistic assumptions, and it directly
addresses "the model says $339,790 saved" sounding falsely precise.
"""

import numpy as np
import pandas as pd

DEFAULT_N_SIMULATIONS = 2000


def simulate_single_run(df: pd.DataFrame, effectiveness_col: str, cost_col: str,
                         churn_col: str = "churn_prob", clv_col: str = "CLV",
                         rng: np.random.Generator = None) -> float:
    """
    One Monte Carlo draw: for each targeted customer, randomly determine
    whether they would have churned, and whether the action (if taken)
    succeeds. Returns total realized value for this single simulated
    outcome:
      - If the customer wouldn't have churned anyway: value = -cost
        (money spent on a customer who didn't need saving).
      - If they would have churned and the action succeeds: value = CLV - cost.
      - If they would have churned and the action fails: value = -cost.
    """
    rng = rng or np.random.default_rng()

    would_churn = rng.uniform(size=len(df)) < df[churn_col].values
    action_succeeds = rng.uniform(size=len(df)) < df[effectiveness_col].values

    saved = would_churn & action_succeeds
    value = np.where(saved, df[clv_col].values, 0) - df[cost_col].values
    return value.sum()


def simulate_revenue_distribution(
    df: pd.DataFrame,
    effectiveness_col: str,
    cost_col: str,
    churn_col: str = "churn_prob",
    clv_col: str = "CLV",
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    seed: int = 42,
) -> dict:
    """
    Runs n_simulations Monte Carlo draws and summarizes the resulting
    distribution of total realized revenue saved: mean, median, and a 95%
    interval (2.5th to 97.5th percentile).

    df should already be filtered down to just the customers being
    targeted (e.g. the top N by expected value under a budget).
    """
    rng = np.random.default_rng(seed)
    results = np.array([
        simulate_single_run(df, effectiveness_col, cost_col, churn_col, clv_col, rng)
        for _ in range(n_simulations)
    ])

    return {
        "mean": results.mean(),
        "median": np.median(results),
        "std": results.std(),
        "ci_lower_95": np.percentile(results, 2.5),
        "ci_upper_95": np.percentile(results, 97.5),
        "n_simulations": n_simulations,
        "raw_results": results,
    }


def main():
    """Quick demo with synthetic data."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "churn_prob": rng.uniform(0.2, 0.8, n),
        "CLV": rng.uniform(500, 9000, n),
    })
    df["effectiveness"] = 0.35  # e.g. discount effectiveness assumption
    df["cost"] = 60.0

    summary = simulate_revenue_distribution(df, "effectiveness", "cost")
    print(f"Simulations: {summary['n_simulations']}")
    print(f"Mean expected revenue saved: ${summary['mean']:,.2f}")
    print(f"95% confidence interval: ${summary['ci_lower_95']:,.2f} - ${summary['ci_upper_95']:,.2f}")
    print(f"Standard deviation: ${summary['std']:,.2f}")


if __name__ == "__main__":
    main()