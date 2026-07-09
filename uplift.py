"""
uplift.py

TRUE CAUSAL UPLIFT MODELING — a step beyond simulate.py.

simulate.py answers: "is this customer worth targeting?" using a FIXED
assumed effectiveness rate for every customer (e.g., "discounts reduce
churn by 35% for everyone"). That's a reasonable starting point, but it's
not individualized — it assumes the discount works the same for a
5-year loyal customer and a 2-month-old month-to-month customer.

uplift.py answers a sharper question: "how much does THIS SPECIFIC
customer's churn probability change if we give them a discount?" This is
the actual question uplift modeling exists to answer, and different
customers can have very different (even negative — sometimes an
intervention backfires) treatment effects.

APPROACH: T-Learner (Two-Model Learner)
  1. Simulate a treatment/control experiment: half of customers are
     randomly given the retention discount, half aren't (in reality this
     would come from a real historical A/B test; here we simulate one
     with a known, heterogeneous treatment effect so we can verify the
     model actually recovers it).
  2. Train two separate churn models: one on the treated group, one on
     the control group.
  3. For every customer, predict churn probability under BOTH models.
     Uplift = P(churn | control) - P(churn | treated). Positive uplift
     means the discount helps THIS customer; near-zero or negative means
     it doesn't (or could even backfire).
  4. Evaluate with a Qini curve — the standard way to measure whether an
     uplift model actually ranks customers better than random targeting.

Run:
    python src/uplift.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

IN_PATH = "data/featured_telco.csv"
OUT_CSV = "data/uplift_scores.csv"
QINI_CHART_OUT = "data/qini_curve.png"

NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "service_count", "price_per_service", "high_friction_payment",
]
CATEGORICAL_FEATURES = [
    "Contract", "InternetService", "PaymentMethod", "tenure_bucket",
]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_SEED = 42


def simulate_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates a retention discount A/B test with a HETEROGENEOUS
    treatment effect: month-to-month, price-sensitive customers benefit
    most from a discount; long-tenure customers barely respond (they were
    staying anyway); a small slice of customers are actually made slightly
    MORE likely to churn (e.g., a discount offer can signal "this company
    is struggling" or draw attention to price for otherwise-unbothered
    customers) — a realistic touch most tutorial uplift examples skip.

    In a real company you would NOT simulate this — you'd use the results
    of an actual historical A/B test. This simulation exists so we can
    verify (below) that the T-learner actually recovers a treatment effect
    that resembles the true one we baked in.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    df = df.copy()

    # Random treatment assignment (as in a real randomized A/B test)
    df["treatment"] = rng.integers(0, 2, size=len(df))

    # Baseline (control) churn probability proxy from observed features
    base_risk = (
        0.35 * (df["Contract"] == "Month-to-month").astype(int)
        + 0.15 * (df["high_friction_payment"])
        + 0.20 * (df["tenure"] < 12).astype(int)
        + 0.10 * (df["price_per_service"] / df["price_per_service"].max())
        + rng.normal(0, 0.05, size=len(df))
    )
    base_risk = np.clip(base_risk, 0.02, 0.95)

    # True heterogeneous treatment effect (this is what the model needs to recover)
    true_uplift = (
        0.25 * (df["Contract"] == "Month-to-month").astype(int)
        + 0.15 * (df["price_per_service"] / df["price_per_service"].max())
        - 0.20 * (df["tenure"] > 48).astype(int)   # loyal customers: discount barely matters
        - 0.05 * (df["service_count"] >= 4).astype(int)  # already-bundled customers: slight backfire
    )
    df["_true_uplift"] = true_uplift  # kept for validation only, not used in training

    treated_risk = np.clip(base_risk - true_uplift, 0.01, 0.98)
    churn_prob = np.where(df["treatment"] == 1, treated_risk, base_risk)
    df["simulated_churn"] = (rng.uniform(size=len(df)) < churn_prob).astype(int)

    return df


def train_t_learner(df: pd.DataFrame):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    treated = df[df["treatment"] == 1]
    control = df[df["treatment"] == 0]

    model_treated = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model_treated.fit(treated[FEATURE_COLS], treated["simulated_churn"])

    # Reuse an identically-structured but separately-fit preprocessor for control
    # (fitting each model's own preprocessor keeps the two models fully independent,
    # which is the correct T-learner setup)
    preprocessor_control = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model_control = Pipeline([
        ("prep", preprocessor_control),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model_control.fit(control[FEATURE_COLS], control["simulated_churn"])

    return model_treated, model_control


def compute_uplift(df: pd.DataFrame, model_treated, model_control) -> pd.DataFrame:
    df = df.copy()
    p_treated = model_treated.predict_proba(df[FEATURE_COLS])[:, 1]
    p_control = model_control.predict_proba(df[FEATURE_COLS])[:, 1]
    df["p_churn_if_treated"] = p_treated
    df["p_churn_if_control"] = p_control
    df["uplift_score"] = p_control - p_treated  # positive = discount helps
    return df


def plot_qini_curve(df: pd.DataFrame):
    """
    Qini curve: sort customers by predicted uplift, then walk down the
    list cumulatively measuring actual observed incremental churn
    reduction (treated churn avoided minus control churn avoided) vs.
    what you'd get from random targeting. A curve that bows above the
    diagonal means the model is successfully finding the customers who
    actually benefit most from treatment — not just the highest-risk ones.
    """
    df = df.sort_values("uplift_score", ascending=False).reset_index(drop=True)
    n = len(df)

    cum_treated_n = np.cumsum(df["treatment"])
    cum_control_n = np.cumsum(1 - df["treatment"])
    cum_treated_churn = np.cumsum(df["treatment"] * df["simulated_churn"])
    cum_control_churn = np.cumsum((1 - df["treatment"]) * df["simulated_churn"])

    # Qini value at each point: incremental churns avoided, scaled to equal group sizes
    with np.errstate(divide="ignore", invalid="ignore"):
        qini = cum_treated_churn - cum_control_churn * (cum_treated_n / cum_control_n.replace(0, np.nan))
    qini = qini.fillna(0)

    pct = np.arange(1, n + 1) / n
    random_line = qini.iloc[-1] * pct

    plt.figure(figsize=(7, 5))
    plt.plot(pct, qini, label="Uplift model (T-learner)")
    plt.plot(pct, random_line, linestyle="--", color="gray", label="Random targeting")
    plt.xlabel("Fraction of customers targeted (ranked by predicted uplift)")
    plt.ylabel("Cumulative incremental churns avoided")
    plt.title("Qini Curve: Uplift Model vs. Random Targeting")
    plt.legend()
    plt.tight_layout()
    plt.savefig(QINI_CHART_OUT, dpi=150)
    plt.close()
    print(f"Qini curve saved to {QINI_CHART_OUT}")


def main():
    df = pd.read_csv(IN_PATH)
    df = simulate_experiment(df)

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)

    model_treated, model_control = train_t_learner(train_df)
    test_df = compute_uplift(test_df, model_treated, model_control)

    # Validation: does predicted uplift correlate with the TRUE uplift we
    # baked into the simulation? (Only possible because this is simulated
    # data — in a real deployment you would NOT have ground truth here,
    # which is exactly why the Qini curve above is the real evaluation tool.)
    corr = np.corrcoef(test_df["uplift_score"], test_df["_true_uplift"])[0, 1]
    print(f"Correlation between predicted and true simulated uplift: {corr:.3f}")
    print("(This validation step only works because we simulated the ground "
          "truth — with real data you'd rely on the Qini curve instead.)\n")

    print("Top 10 customers by predicted uplift (best candidates for discount):")
    print(test_df.sort_values("uplift_score", ascending=False)
          [["tenure", "Contract", "price_per_service", "uplift_score"]].head(10)
          .to_string(index=False))

    print(f"\nCustomers with NEGATIVE predicted uplift (discount may backfire): "
          f"{(test_df['uplift_score'] < 0).sum()} of {len(test_df)}")

    plot_qini_curve(test_df)

    test_df.to_csv(OUT_CSV, index=False)
    print(f"\nUplift scores saved to {OUT_CSV}")


if __name__ == "__main__":
    main()