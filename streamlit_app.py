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
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
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

            # --- Per-customer explanation (SHAP) ---
            # Answers "why did the model flag THIS customer?" instead of
            # just handing back a probability. This is what turns the tool
            # from a scorer into something a retention rep could actually
            # use to have an informed conversation with a customer.
            st.subheader("Explain a specific customer's prediction")

            @st.cache_resource
            def get_explainer(_model, _X_ref):
                preprocessor = _model.named_steps["prep"]
                classifier = _model.named_steps["clf"]
                X_ref_transformed = preprocessor.transform(_X_ref)
                if hasattr(X_ref_transformed, "toarray"):
                    X_ref_transformed = X_ref_transformed.toarray()
                num_names = [c for c in feature_cols if c not in
                             ["Contract", "InternetService", "PaymentMethod", "tenure_bucket"]]
                cat_encoder = preprocessor.named_transformers_["cat"]
                cat_cols = ["Contract", "InternetService", "PaymentMethod", "tenure_bucket"]
                cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
                names = num_names + cat_names
                explainer = shap.TreeExplainer(classifier)
                return explainer, names

            try:
                explainer, transformed_names = get_explainer(model, df[feature_cols])

                display_options = smart_targets.head(50).index.tolist()
                if display_options:
                    chosen_idx = st.selectbox(
                        "Pick a customer from the target list above to explain "
                        "(showing top 50 by expected value):",
                        options=display_options,
                        format_func=lambda i: (
                            f"Row {i} — churn prob {df.loc[i, 'churn_prob']:.0%}, "
                            f"CLV ${df.loc[i, 'CLV']:,.0f}, "
                            f"action: {df.loc[i, 'recommended_action']}"
                        ),
                    )

                    preprocessor = model.named_steps["prep"]
                    row_transformed = preprocessor.transform(df.loc[[chosen_idx], feature_cols])
                    if hasattr(row_transformed, "toarray"):
                        row_transformed = row_transformed.toarray()

                    shap_vals = explainer.shap_values(row_transformed)
                    row_shap = shap_vals[0]

                    contrib_df = pd.DataFrame({
                        "feature": transformed_names,
                        "impact": row_shap,
                    })
                    contrib_df["abs_impact"] = contrib_df["impact"].abs()
                    contrib_df = contrib_df.sort_values("abs_impact", ascending=False).head(10)
                    contrib_df["direction"] = contrib_df["impact"].apply(
                        lambda v: "increases churn risk" if v > 0 else "decreases churn risk"
                    )

                    fig, ax = plt.subplots(figsize=(8, 5))
                    colors = ["#d62728" if v > 0 else "#2ca02c" for v in contrib_df["impact"]]
                    ax.barh(contrib_df["feature"][::-1], contrib_df["impact"][::-1], color=colors[::-1])
                    ax.set_xlabel("Impact on churn probability (SHAP value)")
                    ax.set_title(f"Top factors for customer at row {chosen_idx}")
                    ax.axvline(0, color="gray", linewidth=0.8)
                    st.pyplot(fig)

                    st.caption(
                        "Red bars push this customer's churn probability up; "
                        "green bars pull it down. This is what a retention rep "
                        "could point to when deciding how to approach the conversation."
                    )
            except Exception as e:
                st.warning(f"Couldn't generate explanation for this dataset: {e}")
    else:
        st.warning(
            "No trained model found at data/model.joblib. "
            "Run `python src/train.py` first."
        )
else:
    st.info("Upload a CSV to get started. Run the pipeline in src/ first to generate one.")