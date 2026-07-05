"""
streamlit_app.py

Deployable demo: upload a customer CSV, get back a ranked list of who to
target for retention and with which action, plus total expected value saved.

This is the piece that turns your project from "a notebook" into
"a tool a business could actually use" — deploy this on Streamlit
Community Cloud or Hugging Face Spaces and link it on your resume/LinkedIn.

Run locally:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

st.set_page_config(page_title="Churn Intervention Planner", layout="wide")

st.title("Churn Intervention Planner")
st.markdown(
    """
    Upload a customer dataset to get back a ranked action plan:
    who's likely to churn, what it's worth to save them, and which
    retention action gives the best expected return.
    """
)

MODEL_PATH = "data/model.joblib"

uploaded_file = st.file_uploader("Upload enriched + featured customer CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df)} customers.")

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)

        feature_cols = [
            "tenure", "MonthlyCharges", "TotalCharges", "service_count",
            "price_per_service", "high_friction_payment",
            "Contract", "InternetService", "PaymentMethod", "tenure_bucket",
        ]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.error(f"Missing expected columns: {missing}")
        else:
            df["churn_prob"] = model.predict_proba(df[feature_cols])[:, 1]

            # Expected value calc (mirrors simulate.py)
            DISCOUNT_EFFECTIVENESS = 0.35
            OUTREACH_EFFECTIVENESS = 0.15

            df["ev_discount"] = (
                df["churn_prob"] * DISCOUNT_EFFECTIVENESS * df["CLV"] - df["cost_discount"]
            )
            df["ev_outreach"] = (
                df["churn_prob"] * OUTREACH_EFFECTIVENESS * df["CLV"] - df["cost_outreach"]
            )

            def pick_action(row):
                options = {"discount": row["ev_discount"], "outreach": row["ev_outreach"]}
                best_action = max(options, key=options.get)
                best_value = options[best_action]
                if best_value <= 0:
                    return pd.Series(["no_action", 0.0])
                return pd.Series([best_action, best_value])

            df[["recommended_action", "expected_value"]] = df.apply(pick_action, axis=1)
            df = df.sort_values("expected_value", ascending=False)

            # --- Budget constraint ---
            # Without a budget cap, "value-based" and "naive churn-probability"
            # targeting tend to converge, since almost every customer looks
            # profitable to target. The real case for value-based targeting
            # shows up when you can't target everyone — this slider makes
            # that constraint explicit and shows the resulting gap.
            st.subheader("Retention budget")
            max_budget = len(df)
            default_budget = min(200, max_budget)
            budget = st.slider(
                "How many customers can you afford to target this cycle?",
                min_value=1,
                max_value=max_budget,
                value=default_budget,
                step=1,
                help="Personalized retention outreach doesn't scale infinitely — "
                     "this simulates a realistic cap on how many customers your "
                     "team can actually act on.",
            )

            smart_targets = df.sort_values("expected_value", ascending=False).head(budget)
            smart_ev = smart_targets["expected_value"].sum()

            naive_targets = df.sort_values("churn_prob", ascending=False).head(budget)
            naive_ev = naive_targets["expected_value"].sum()

            improvement = smart_ev - naive_ev
            pct_improvement = (improvement / naive_ev * 100) if naive_ev != 0 else 0.0
            overlap = len(set(smart_targets.index) & set(naive_targets.index))
            overlap_pct = overlap / budget * 100 if budget > 0 else 0.0

            col1, col2, col3 = st.columns(3)
            col1.metric("Value-based targeting", f"${smart_ev:,.0f}")
            col2.metric(
                "Naive (churn prob only)", f"${naive_ev:,.0f}",
                delta=f"{pct_improvement:+.1f}% vs. value-based" if naive_ev else None,
                delta_color="inverse",
            )
            col3.metric("Target list overlap", f"{overlap_pct:.0f}%",
                        help="How many of the same customers both strategies would target. "
                             "Lower overlap means the two rankings disagree more — and "
                             "that disagreement is where value-based targeting earns its keep.")

            if pct_improvement > 0:
                st.success(
                    f"At a budget of {budget} customers, ranking by expected value "
                    f"captures **{pct_improvement:.1f}% more** revenue than ranking by "
                    f"churn probability alone — by choosing higher-CLV customers the "
                    f"naive approach would have missed."
                )
            elif budget >= max_budget * 0.9:
                st.info(
                    "At this budget size, nearly every customer is being targeted "
                    "under both strategies, so they converge. Try a smaller budget "
                    "to see the gap open up."
                )

            st.subheader(f"Ranked action plan (top {budget} by expected value)")
            display_cols = [
                "churn_prob", "CLV", "recommended_action", "expected_value"
            ]
            display_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(
                smart_targets[display_cols].style.format({
                    "churn_prob": "{:.1%}",
                    "CLV": "${:,.0f}",
                    "expected_value": "${:,.0f}",
                }),
                use_container_width=True,
            )

            st.download_button(
                "Download this action plan as CSV",
                smart_targets.to_csv(index=False),
                "intervention_plan.csv",
                "text/csv",
            )
    else:
        st.warning(
            "No trained model found at data/model.joblib. "
            "Run `python src/train.py` first."
        )
else:
    st.info("Upload a CSV to get started. Run the pipeline in src/ first to generate one.")