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

            total_ev = df["expected_value"].sum()
            n_targeted = (df["recommended_action"] != "no_action").sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Customers evaluated", len(df))
            col2.metric("Worth targeting", n_targeted)
            col3.metric("Total expected value saved", f"${total_ev:,.0f}")

            st.subheader("Ranked action plan")
            display_cols = [
                "churn_prob", "CLV", "recommended_action", "expected_value"
            ]
            display_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(
                df[display_cols].style.format({
                    "churn_prob": "{:.1%}",
                    "CLV": "${:,.0f}",
                    "expected_value": "${:,.0f}",
                }),
                use_container_width=True,
            )

            st.download_button(
                "Download full action plan as CSV",
                df.to_csv(index=False),
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
