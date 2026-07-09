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
from groq import Groq

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from genai_agent import ask_question, generate_outreach_message
from action_optimizer import add_action_recommendations, action_menu_summary, ACTIONS

st.set_page_config(page_title="Churn Intervention Planner", layout="wide")

st.title("Churn Intervention Planner")
st.markdown(
    """
    Upload a customer dataset to get back a ranked action plan:
    who's likely to churn, what it's worth to save them, and which
    retention action gives the best expected return.
    """
)

# --- GenAI agent setup (optional) ---
# Free via Groq (https://console.groq.com/keys) — no billing required.
# Key is only held in this session's memory, never written to disk.
with st.sidebar:
    st.header("GenAI agent (optional)")
    st.caption(
        "Powers the Q&A agent and outreach message generator below. "
        "Free tier — get a key at console.groq.com/keys"
    )
    groq_api_key = st.text_input(
        "Groq API key", type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Never stored — only kept in memory for this session.",
    )

groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

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

            # --- Multi-action optimizer ---
            # Instead of a hardcoded binary choice (discount vs. outreach),
            # evaluate a full menu of retention actions and pick whichever
            # has the best expected return for each customer. See
            # src/action_optimizer.py for the menu and the two optimization
            # modes (max total value vs. max ROI per dollar spent).
            st.subheader("Retention action menu")
            st.caption(
                "Costs and effectiveness rates below are assumptions — replace "
                "with real historical campaign data when available."
            )
            st.dataframe(action_menu_summary(), use_container_width=True, hide_index=True)

            opt_mode_label = st.radio(
                "Optimization goal",
                options=["Maximize total value per customer", "Maximize ROI (return per dollar spent)"],
                help=(
                    "Maximize total value: best when each customer's action is funded "
                    "independently (no shared budget). "
                    "Maximize ROI: best under a limited shared budget, since cheap, "
                    "efficient actions let you afford to treat more customers overall."
                ),
                horizontal=True,
            )
            opt_mode = "max_value" if opt_mode_label.startswith("Maximize total") else "max_roi"

            df = add_action_recommendations(df, ACTIONS, optimization_mode=opt_mode)
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

            # --- Diminishing returns chart ---
            # A single budget number doesn't show the shape of the tradeoff.
            # This chart shows expected revenue saved across a full range of
            # budget sizes, so a manager can see the diminishing-returns
            # curve directly instead of one point on it.
            st.subheader("Revenue saved vs. customers targeted")
            checkpoints = sorted(set(
                [int(max_budget * p) for p in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]]
                + [budget]
            ))
            checkpoints = [c for c in checkpoints if c > 0]

            curve_df = pd.DataFrame({
                "customers_targeted": checkpoints,
                "revenue_saved": [
                    df.sort_values("expected_value", ascending=False).head(c)["expected_value"].sum()
                    for c in checkpoints
                ],
            })

            fig_curve, ax_curve = plt.subplots(figsize=(9, 4))
            ax_curve.plot(curve_df["customers_targeted"], curve_df["revenue_saved"], marker="o")
            ax_curve.axvline(budget, color="gray", linestyle="--", alpha=0.6, label=f"Current budget ({budget})")
            ax_curve.set_xlabel("Customers targeted")
            ax_curve.set_ylabel("Expected revenue saved ($)")
            ax_curve.set_title("Diminishing Returns: More Budget Helps, But Less at the Margin")
            ax_curve.legend()
            ax_curve.grid(alpha=0.3)
            st.pyplot(fig_curve)
            st.caption(
                "The curve flattens as budget grows — each additional customer "
                "targeted contributes less than the last, since customers are "
                "targeted in order of expected value (highest first)."
            )

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

                    # --- Outreach message generator ---
                    st.subheader("Draft a retention email for this customer")
                    if groq_client is None:
                        st.info("Enter a free Groq API key in the sidebar to generate a message.")
                    else:
                        if st.button("Generate outreach email"):
                            with st.spinner("Drafting message..."):
                                top_factors = [
                                    (row["feature"], row["direction"])
                                    for _, row in contrib_df.head(3).iterrows()
                                ]
                                customer_dict = df.loc[chosen_idx].to_dict()
                                action = df.loc[chosen_idx, "recommended_action"]
                                try:
                                    email_text = generate_outreach_message(
                                        customer_dict, top_factors, action, groq_client
                                    )
                                    st.text_area("Draft email", email_text, height=250)
                                except Exception as e:
                                    st.error(f"Couldn't generate message: {e}")
            except Exception as e:
                st.warning(f"Couldn't generate explanation for this dataset: {e}")

            # --- Q&A agent ---
            # Real tool-calling: the model decides which aggregation to run
            # against the actual uploaded dataframe, and answers using the
            # real numbers returned — not a guess from training data.
            st.subheader("Ask a question about this data")
            if groq_client is None:
                st.info("Enter a free Groq API key in the sidebar to use the Q&A agent.")
            else:
                example_qs = (
                    "Try: \"Which contract type has the highest churn rate?\" or "
                    "\"What's the average CLV for customers with fiber internet?\""
                )
                st.caption(example_qs)
                question = st.text_input("Your question")
                if st.button("Ask") and question:
                    with st.spinner("Thinking..."):
                        try:
                            answer = ask_question(question, df, groq_client)
                            st.write(answer)
                        except Exception as e:
                            st.error(f"Couldn't answer that: {e}")
    else:
        st.warning(
            "No trained model found at data/model.joblib. "
            "Run `python src/train.py` first."
        )
else:
    st.info("Upload a CSV to get started. Run the pipeline in src/ first to generate one.")