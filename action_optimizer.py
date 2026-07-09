"""
action_optimizer.py

Generalizes simulate.py's binary "discount vs. outreach" choice into a
proper multi-action optimizer: given ANY number of retention actions with
different costs and effectiveness rates, pick whichever has the highest
expected ROI for each customer.

This is the difference between:
  "Predict churn" -> "here's who's at risk"
and
  "Recommend the optimal intervention" -> "here's who's at risk, AND
  here's the single best action to take, chosen from a real menu of
  options, not just one hardcoded lever."

Action effectiveness rates below are ASSUMPTIONS (clearly labeled as such)
-- in a real deployment these would come from historical campaign data
(e.g. "past phone-call campaigns retained 35% of contacted at-risk
customers"). The point of this module is the OPTIMIZATION LOGIC, which
works identically regardless of where the effectiveness numbers come from.
"""

import pandas as pd

# --- Action menu: cost and effectiveness (probability of preventing churn) ---
# Effectiveness = probability THIS action, if taken, prevents the churn that
# was otherwise going to happen. Costs and rates here are illustrative
# assumptions; swap in real historical campaign data when available.
ACTIONS = {
    "email": {"cost": 2.0, "effectiveness": 0.05},
    "sms": {"cost": 5.0, "effectiveness": 0.12},
    "phone_call": {"cost": 20.0, "effectiveness": 0.35},
    "discount": {"cost": 60.0, "effectiveness": 0.70},
}


def compute_action_value(churn_prob: float, clv: float, cost: float, effectiveness: float) -> dict:
    """
    Expected value of taking this action on a customer with this churn
    probability and CLV:
        expected_value = (churn_prob * effectiveness * CLV) - cost
    ROI = expected_value / cost (return per dollar spent on this action).
    """
    expected_value = (churn_prob * effectiveness * clv) - cost
    roi = (expected_value / cost) if cost > 0 else None
    return {"expected_value": expected_value, "roi": roi}


def recommend_best_action(churn_prob: float, clv: float, actions: dict = None,
                           optimization_mode: str = "max_value") -> dict:
    """
    Evaluates every action in the menu for one customer and returns the
    best one.

    optimization_mode:
      - "max_value" (default): pick the action with the highest absolute
        expected value. Correct when there's effectively no budget
        constraint (e.g. each customer's action is funded independently).
      - "max_roi": pick the action with the highest return per dollar
        spent. This matters when there's a SHARED, LIMITED budget across
        many customers — in that case, cheap-and-efficient actions (e.g.
        email) let you afford to treat more customers overall, even
        though any single customer's raw expected value looks smaller
        than what a discount would produce. This is the same "budget vs.
        no budget" distinction budget_sensitivity.py explores at the
        portfolio level, applied here at the per-action level.

    Returns "no_action" if every action would be a net loss.
    """
    actions = actions or ACTIONS
    results = {}
    for name, params in actions.items():
        results[name] = compute_action_value(churn_prob, clv, params["cost"], params["effectiveness"])

    if optimization_mode == "max_roi":
        # Only consider actions with positive expected value when ranking by ROI —
        # otherwise a barely-negative, near-zero-cost action can look like a great
        # "ROI" purely from a small denominator.
        positive_ev = {k: v for k, v in results.items() if v["expected_value"] > 0}
        if not positive_ev:
            return {
                "recommended_action": "no_action", "expected_value": 0.0,
                "roi": None, "all_options": results,
            }
        best_action = max(positive_ev, key=lambda name: positive_ev[name]["roi"])
    else:
        best_action = max(results, key=lambda name: results[name]["expected_value"])

    best = results[best_action]

    if best["expected_value"] <= 0:
        return {
            "recommended_action": "no_action",
            "expected_value": 0.0,
            "roi": None,
            "all_options": results,
        }

    return {
        "recommended_action": best_action,
        "expected_value": best["expected_value"],
        "roi": best["roi"],
        "all_options": results,
    }


def add_action_recommendations(df: pd.DataFrame, actions: dict = None,
                                churn_col: str = "churn_prob", clv_col: str = "CLV",
                                optimization_mode: str = "max_value") -> pd.DataFrame:
    """
    Vectorized-ish wrapper: applies recommend_best_action to every row of a
    dataframe and adds recommended_action / expected_value / roi columns.
    """
    actions = actions or ACTIONS
    df = df.copy()

    recommendations = df.apply(
        lambda row: recommend_best_action(row[churn_col], row[clv_col], actions, optimization_mode),
        axis=1,
    )

    df["recommended_action"] = recommendations.apply(lambda r: r["recommended_action"])
    df["expected_value"] = recommendations.apply(lambda r: r["expected_value"])
    df["roi"] = recommendations.apply(lambda r: r["roi"])

    return df


def action_menu_summary(actions: dict = None) -> pd.DataFrame:
    """Returns the action menu as a readable dataframe, e.g. for display in the app."""
    actions = actions or ACTIONS
    return pd.DataFrame([
        {"action": name, "cost": params["cost"], "effectiveness": params["effectiveness"]}
        for name, params in actions.items()
    ])


def main():
    """Quick demo against a few example customers."""
    examples = [
        {"name": "High CLV, high churn risk", "churn_prob": 0.7, "clv": 9000},
        {"name": "Low CLV, high churn risk", "churn_prob": 0.7, "clv": 300},
        {"name": "High CLV, low churn risk", "churn_prob": 0.1, "clv": 9000},
    ]

    print("Action menu:")
    print(action_menu_summary().to_string(index=False))
    print()

    for ex in examples:
        result_value = recommend_best_action(ex["churn_prob"], ex["clv"], optimization_mode="max_value")
        result_roi = recommend_best_action(ex["churn_prob"], ex["clv"], optimization_mode="max_roi")
        print(f"{ex['name']} (churn_prob={ex['churn_prob']}, CLV=${ex['clv']}):")
        print(f"  max_value mode -> {result_value['recommended_action']} "
              f"(EV: ${result_value['expected_value']:.2f})")
        print(f"  max_roi mode   -> {result_roi['recommended_action']} "
              f"(EV: ${result_roi['expected_value']:.2f}, ROI: {result_roi['roi']:.2f})")
        for action_name, action_result in result_value["all_options"].items():
            roi_str = f", ROI={action_result['roi']:.2f}" if action_result['roi'] is not None else ""
            print(f"    {action_name}: EV=${action_result['expected_value']:.2f}{roi_str}")
        print()


if __name__ == "__main__":
    main()