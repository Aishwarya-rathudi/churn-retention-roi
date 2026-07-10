"""
streamlit_app.py

Deployable demo: upload a customer CSV, get back a ranked list of who to
target for retention and with which action, plus total expected value saved.

Organized into tabs:
  - Overview: dataset summary
  - Churn Prediction: model output, risk distribution
  - ROI Optimizer: action menu, budget slider, diminishing returns, confidence interval
  - Explainability: customer summary card + SHAP explanation
  - Retention Copilot: Q&A agent + outreach message generator
  - Model Performance: ROC/PR/calibration (if evaluate.py has been run)

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
from action_optimizer import add_action_recommendations, action_menu_summary, ACTIONS, recommend_best_action
from uncertainty import simulate_revenue_distribution

st.set_page_config(page_title="Churn Intervention Planner", layout="wide")

st.title("Churn Intervention Planner")
st.markdown(
    """
    Upload a customer dataset to get a ranked action plan: who's likely to
    churn, what it's worth to save them, and which retention action gives
    the best expected return.
    """
)

# --- GenAI agent setup (optional) ---
# Free via Groq (https://console.groq.com/keys) — no billing required.
# Key is only held in this session's memory, never written to disk.
with st.sidebar:
    st.header("GenAI agent (optional)")
    st.caption(
        "Powers the Q&A agent and outreach message generator. "
        "Free tier — get a key at console.groq.com/keys"
    )
    groq_api_key = st.text_input(
        "Groq API key", type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Never stored — only kept in memory for this session.",
    )

groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

MODEL_PATH = "data/model.joblib"
METRICS_PATH = "data/model_metrics.csv"
EVAL_CHART_PATH = "data/model_evaluation.png"

FEATURE_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges", "service_count",
    "price_per_service", "high_friction_payment",
    "Contract", "InternetService", "PaymentMethod", "tenure_bucket",
]

uploaded_file = st.file_uploader("Upload enriched + featured customer CSV", type="csv")

if uploaded_file is None:
    st.info("Upload a CSV to get started. Run the pipeline in src/ first to generate one.")
elif not os.path.exists(MODEL_PATH):
    st.warning("No trained model found at data/model.joblib. Run `python src/train.py` first.")
else:
    df = pd.read_csv(uploaded_file)
    model = joblib.load(MODEL_PATH)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing expected columns: {missing}")
    else:
        df["churn_prob"] = model.predict_proba(df[FEATURE_COLS])[:, 1]

        tab_overview, tab_churn, tab_roi, tab_explain, tab_agent, tab_monitor = st.tabs([
            "Overview", "Churn Prediction", "ROI Optimizer",
            "Why This Customer?", "Retention Copilot", "Model Performance",
        ])

        # =========================================================
        # TAB: OVERVIEW
        # =========================================================
        with tab_overview:
            st.subheader("Dataset summary")
            ov_col1, ov_col2, ov_col3 = st.columns(3)
            ov_col1.metric("Customers loaded", f"{len(df):,}")
            ov_col2.metric("Average churn probability", f"{df['churn_prob'].mean():.1%}")
            if "CLV" in df.columns:
                ov_col3.metric("Average CLV", f"${df['CLV'].mean():,.0f}")

            st.markdown(
                """
                **How to use this app:**
                1. **Churn Prediction** — see the model's raw risk scores
                2. **ROI Optimizer** — pick a retention action strategy and a budget
                3. **Why This Customer?** — see why any individual customer was flagged
                4. **Retention Copilot** — ask questions or draft outreach messages
                5. **Model Performance** — technical evaluation metrics
                """
            )

            if "Contract" in df.columns:
                st.subheader("Churn rate by contract type")
                contract_churn = df.groupby("Contract")["churn_prob"].mean().sort_values(ascending=False)
                fig_ov, ax_ov = plt.subplots(figsize=(7, 3.5))
                ax_ov.bar(contract_churn.index, contract_churn.values, color="#4C72B0")
                ax_ov.set_ylabel("Average predicted churn probability")
                ax_ov.set_title("Churn Risk by Contract Type")
                st.pyplot(fig_ov)

        # =========================================================
        # TAB: CHURN PREDICTION
        # =========================================================
        with tab_churn:
            st.subheader("Churn probability distribution")
            fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
            ax_dist.hist(df["churn_prob"], bins=30, color="#C44E52", edgecolor="white")
            ax_dist.set_xlabel("Predicted churn probability")
            ax_dist.set_ylabel("Number of customers")
            ax_dist.set_title("Distribution of Predicted Churn Risk")
            st.pyplot(fig_dist)

            st.subheader("Highest-risk customers (by churn probability alone)")
            risk_cols = [c for c in ["tenure", "Contract", "MonthlyCharges", "CLV", "churn_prob"] if c in df.columns]
            top_risk = df.sort_values("churn_prob", ascending=False).head(20)[risk_cols]
            st.dataframe(
                top_risk.style.format({"churn_prob": "{:.1%}", "CLV": "${:,.0f}", "MonthlyCharges": "${:,.0f}"}),
                width='stretch',
            )
            st.caption(
                "Note: this ranks by raw churn risk only. See the ROI Optimizer tab for "
                "who's actually worth targeting once cost and value are factored in."
            )

        # =========================================================
        # TAB: ROI OPTIMIZER
        # =========================================================
        with tab_roi:
            st.subheader("Retention action menu")
            st.caption(
                "Costs and effectiveness rates below are assumptions — adjust them "
                "below to run your own scenario, or replace with real historical "
                "campaign data when available."
            )

            # --- Scenario planning ---
            # Lets you change cost/effectiveness assumptions (and see the
            # optimizer respond instantly) instead of only viewing one fixed
            # scenario. Turns this from a static report into a planning tool.
            with st.expander("Scenario planning: adjust cost & effectiveness assumptions", expanded=False):
                st.caption(
                    "Changes here immediately recompute recommended actions, "
                    "budget metrics, and charts below."
                )
                scenario_actions = {}
                for action_name, params in ACTIONS.items():
                    sp_col1, sp_col2 = st.columns(2)
                    new_cost = sp_col1.number_input(
                        f"{action_name} — cost ($)", min_value=0.0,
                        value=float(params["cost"]), step=1.0, key=f"cost_{action_name}",
                    )
                    new_eff = sp_col2.slider(
                        f"{action_name} — effectiveness", min_value=0.0, max_value=1.0,
                        value=float(params["effectiveness"]), step=0.01, key=f"eff_{action_name}",
                    )
                    scenario_actions[action_name] = {"cost": new_cost, "effectiveness": new_eff}
                if st.button("Reset to default assumptions"):
                    for action_name, params in ACTIONS.items():
                        st.session_state[f"cost_{action_name}"] = float(params["cost"])
                        st.session_state[f"eff_{action_name}"] = float(params["effectiveness"])
                    st.rerun()

            active_actions = scenario_actions
            st.dataframe(action_menu_summary(active_actions), width='stretch', hide_index=True)

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

            df = add_action_recommendations(df, active_actions, optimization_mode=opt_mode)
            df = df.sort_values("expected_value", ascending=False)

            # --- Budget constraint ---
            st.subheader("Retention budget")
            max_budget = len(df)
            default_budget = min(200, max_budget)
            budget = st.slider(
                "How many customers can you afford to target this cycle?",
                min_value=1, max_value=max_budget, value=default_budget, step=1,
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
            col1.metric(
                "Value-based targeting", f"${smart_ev:,.0f}",
                delta=f"{pct_improvement:+.1f}% more than naive" if naive_ev else None,
            )
            col2.metric("Naive (churn prob only)", f"${naive_ev:,.0f}")
            col3.metric("Target list overlap", f"{overlap_pct:.0f}%",
                        help="How many of the same customers both strategies would target. "
                             "Lower overlap means the two rankings disagree more.")

            with st.expander("Show confidence interval (Monte Carlo simulation)"):
                sim_df = smart_targets.copy()
                sim_df["_effectiveness"] = sim_df["recommended_action"].map(
                    lambda a: active_actions.get(a, {}).get("effectiveness", 0.0)
                )
                sim_df["_cost"] = sim_df["recommended_action"].map(
                    lambda a: active_actions.get(a, {}).get("cost", 0.0)
                )
                sim_summary = simulate_revenue_distribution(
                    sim_df, effectiveness_col="_effectiveness", cost_col="_cost",
                )
                ci_col1, ci_col2 = st.columns(2)
                ci_col1.metric("Expected revenue (mean of simulation)", f"${sim_summary['mean']:,.0f}")
                ci_col2.metric(
                    "95% confidence interval",
                    f"${sim_summary['ci_lower_95']:,.0f} – ${sim_summary['ci_upper_95']:,.0f}",
                )
                st.caption(
                    f"Based on {sim_summary['n_simulations']:,} simulated outcomes, accounting for "
                    "both churn uncertainty and retention-action uncertainty."
                )

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
                "targeted contributes less than the last."
            )

            if pct_improvement > 0:
                st.success(
                    f"At a budget of {budget} customers, ranking by expected value "
                    f"captures **{pct_improvement:.1f}% more** revenue than ranking by "
                    f"churn probability alone."
                )
            elif budget >= max_budget * 0.9:
                st.info(
                    "At this budget size, nearly every customer is being targeted "
                    "under both strategies, so they converge. Try a smaller budget."
                )

            st.subheader(f"Ranked action plan (top {budget} by expected value)")
            display_cols = [c for c in ["churn_prob", "CLV", "recommended_action", "expected_value"] if c in df.columns]
            st.dataframe(
                smart_targets[display_cols].style.format({
                    "churn_prob": "{:.1%}", "CLV": "${:,.0f}", "expected_value": "${:,.0f}",
                }),
                width='stretch',
            )
            st.download_button(
                "Download this action plan as CSV",
                smart_targets.to_csv(index=False), "intervention_plan.csv", "text/csv",
            )

        # =========================================================
        # TAB: EXPLAINABILITY
        # =========================================================
        chosen_idx = None
        contrib_df = None
        app_context = None
        explainer = None
        transformed_names = None
        with tab_explain:
            st.subheader("Explain a specific customer's prediction")

            @st.cache_resource
            def get_explainer(_model, _X_ref):
                preprocessor = _model.named_steps["prep"]
                classifier = _model.named_steps["clf"]
                X_ref_transformed = preprocessor.transform(_X_ref)
                if hasattr(X_ref_transformed, "toarray"):
                    X_ref_transformed = X_ref_transformed.toarray()
                num_names = [c for c in FEATURE_COLS if c not in
                             ["Contract", "InternetService", "PaymentMethod", "tenure_bucket"]]
                cat_encoder = preprocessor.named_transformers_["cat"]
                cat_cols = ["Contract", "InternetService", "PaymentMethod", "tenure_bucket"]
                cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
                names = num_names + cat_names
                # shap.Explainer (not TreeExplainer) auto-dispatches to the
                # right algorithm for whichever model won train.py's model
                # comparison — TreeExplainer alone would break if Logistic
                # Regression or another non-tree model was selected.
                background = X_ref_transformed[:min(100, len(X_ref_transformed))]
                explainer = shap.Explainer(classifier, background)
                return explainer, names

            try:
                explainer, transformed_names = get_explainer(model, df[FEATURE_COLS])

                display_options = smart_targets.head(50).index.tolist()
                if display_options:
                    chosen_idx = st.selectbox(
                        "Pick a customer from the target list to explain (top 50 by expected value):",
                        options=display_options,
                        format_func=lambda i: (
                            f"Row {i} — churn prob {df.loc[i, 'churn_prob']:.0%}, "
                            f"CLV ${df.loc[i, 'CLV']:,.0f}, "
                            f"action: {df.loc[i, 'recommended_action']}"
                        ),
                    )

                    preprocessor = model.named_steps["prep"]
                    row_transformed = preprocessor.transform(df.loc[[chosen_idx], FEATURE_COLS])
                    if hasattr(row_transformed, "toarray"):
                        row_transformed = row_transformed.toarray()

                    row_explanation = explainer(row_transformed)
                    row_shap = row_explanation.values[0]

                    contrib_df = pd.DataFrame({"feature": transformed_names, "impact": row_shap})
                    contrib_df["abs_impact"] = contrib_df["impact"].abs()
                    contrib_df = contrib_df.sort_values("abs_impact", ascending=False).head(10)
                    contrib_df["direction"] = contrib_df["impact"].apply(
                        lambda v: "increases churn risk" if v > 0 else "decreases churn risk"
                    )

                    # --- Customer summary card ---
                    # Compact custom HTML instead of st.metric, since metric's
                    # large number display was too visually heavy for a dense
                    # profile card with 8 fields.
                    customer_row = df.loc[chosen_idx]
                    risk_prob = customer_row["churn_prob"]
                    risk_level = "High" if risk_prob >= 0.6 else ("Medium" if risk_prob >= 0.3 else "Low")
                    risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[risk_level]
                    top_reason = contrib_df.iloc[0]["feature"] if len(contrib_df) else "N/A"
                    clv_val = customer_row.get("CLV", None)
                    value_tag = "High-value customer" if (clv_val is not None and clv_val >= df["CLV"].median()) \
                        else "Standard-value customer"

                    def _card_field(label, value):
                        return (
                            f'<div style="padding:4px 0;">'
                            f'<div style="font-size:0.75rem; color:#888; margin-bottom:1px;">{label}</div>'
                            f'<div style="font-size:0.95rem; font-weight:600;">{value}</div>'
                            f'</div>'
                        )

                    fields = [
                        ("Tenure", f"{customer_row.get('tenure', 'N/A')} months"),
                        ("Contract", customer_row.get("Contract", "N/A")),
                        ("Internet", customer_row.get("InternetService", "N/A")),
                        ("Monthly charges", f"${customer_row.get('MonthlyCharges', 0):,.0f}"),
                        ("Churn risk", f"{risk_color} {risk_level} ({risk_prob:.0%})"),
                        ("Lifetime value", f"${clv_val:,.0f}" if clv_val is not None else "N/A"),
                        ("Recommended action", customer_row.get("recommended_action", "N/A")),
                        ("Top risk factor", top_reason),
                    ]

                    card_html = (
                        f'<div style="font-size:1rem; font-weight:600; margin-bottom:6px;">'
                        f'{value_tag} — Row {chosen_idx}</div>'
                        f'<div style="display:grid; grid-template-columns:repeat(4, 1fr); '
                        f'gap:2px 16px; border:1px solid #333; border-radius:8px; padding:10px 14px;">'
                        + "".join(_card_field(label, val) for label, val in fields)
                        + "</div>"
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

                    # --- Action rationale ---
                    # Shows the FULL comparison across all actions for this
                    # customer, not just the winner — makes the recommendation
                    # transparent instead of a black-box label.
                    rationale = recommend_best_action(
                        customer_row["churn_prob"], clv_val if clv_val is not None else 0,
                        active_actions,
                    )
                    st.markdown("**Why this action?**")
                    rationale_rows = []
                    for action_name, opts in rationale["all_options"].items():
                        rationale_rows.append({
                            "action": action_name,
                            "expected_value": opts["expected_value"],
                            "roi": opts["roi"],
                            "chosen": "✅" if action_name == rationale["recommended_action"] else "",
                        })
                    rationale_df = pd.DataFrame(rationale_rows).sort_values("expected_value", ascending=False)
                    st.dataframe(
                        rationale_df.style.format({"expected_value": "${:,.0f}", "roi": "{:.1f}x"}),
                        width='stretch', hide_index=True,
                    )
                    if rationale["recommended_action"] != "no_action":
                        st.caption(
                            f"**{rationale['recommended_action']}** was chosen because it has the "
                            f"highest expected value (${rationale['expected_value']:,.0f}) among "
                            f"available interventions, given this customer's churn probability and CLV."
                        )
                    else:
                        st.caption("No action is recommended — every option would be a net loss for this customer.")

                    # --- Build app-state context for the GenAI agent ---
                    # This is what lets the Q&A agent answer "why was this
                    # customer selected?" grounded in the app's actual current
                    # state, instead of only being able to answer generic
                    # aggregate questions about the whole dataset.
                    top_shap_factors = ", ".join(
                        f"{row['feature']} ({row['direction']})"
                        for _, row in contrib_df.head(3).iterrows()
                    )
                    app_context = (
                        f"Row {chosen_idx} is currently selected in the app. "
                        f"Churn probability: {customer_row['churn_prob']:.0%}. "
                        f"CLV: ${clv_val:,.0f}. "
                        f"Recommended action: {rationale['recommended_action']}. "
                        f"Expected value of that action: ${rationale['expected_value']:,.2f}. "
                        f"Reason the action was chosen: it has the highest expected value "
                        f"among available options given this customer's churn probability and CLV. "
                        f"Top SHAP factors driving this customer's churn risk: {top_shap_factors}."
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
                        "green bars pull it down."
                    )
            except Exception as e:
                st.warning(f"Couldn't generate explanation for this dataset: {e}")

        # =========================================================
        # TAB: GENAI ASSISTANT
        # =========================================================
        with tab_agent:
            st.subheader("Draft a retention email")
            if chosen_idx is None or contrib_df is None:
                st.info('Pick a customer in the "Why This Customer?" tab first.')
            elif groq_client is None:
                st.info("Enter a free Groq API key in the sidebar to generate a message.")
            else:
                st.caption(f'Drafting for customer at row {chosen_idx} (selected in "Why This Customer?" tab).')
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

            st.divider()
            st.subheader("Ask a question about this data")
            if groq_client is None:
                st.info("Enter a free Groq API key in the sidebar to use the Q&A agent.")
            else:
                if app_context:
                    with st.expander("Currently selected customer (used as context for your question)"):
                        st.caption(app_context)
                    st.caption(
                        f"Try: \"Why was row {chosen_idx} selected?\" or "
                        "\"Why wasn't row 5127 selected?\" (any row number works, "
                        "not just the one selected above) or "
                        "\"Which contract type has the highest churn rate?\""
                    )
                else:
                    st.caption(
                        "Try: \"Which contract type has the highest churn rate?\" or "
                        "\"What's the maximum discount I can give a customer worth $9000 "
                        "with 60% retention probability and $40 campaign cost?\" "
                        "(Pick a customer in the \"Why This Customer?\" tab to also ask "
                        "questions about that specific customer.)"
                    )
                question = st.text_input("Your question")
                if st.button("Ask") and question:
                    with st.spinner("Thinking..."):
                        try:
                            answer = ask_question(
                                question, df, groq_client, app_context=app_context, budget=budget,
                                model=model, explainer=explainer,
                                feature_cols=FEATURE_COLS, transformed_names=transformed_names,
                            )
                            st.write(answer)
                        except Exception as e:
                            st.error(f"Couldn't answer that: {e}")

        # =========================================================
        # TAB: MODEL PERFORMANCE
        # =========================================================
        with tab_monitor:
            st.subheader("Model evaluation metrics")
            if os.path.exists(METRICS_PATH):
                metrics_df = pd.read_csv(METRICS_PATH)
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("ROC-AUC", f"{metrics_df['roc_auc'].iloc[0]:.3f}")
                m_col2.metric("PR-AUC", f"{metrics_df['pr_auc'].iloc[0]:.3f}")
                m_col3.metric("Precision", f"{metrics_df['precision'].iloc[0]:.3f}")
                m_col4.metric("Recall", f"{metrics_df['recall'].iloc[0]:.3f}")
            else:
                st.info("Run `python src/evaluate.py` to generate model performance metrics.")

            if os.path.exists(EVAL_CHART_PATH):
                st.image(EVAL_CHART_PATH, caption="ROC curve, Precision-Recall curve, and calibration curve")
                st.caption(
                    "The calibration curve (right panel) checks whether a customer with "
                    "'70% predicted churn probability' actually churns about 70% of the "
                    "time — important here since churn probability is directly multiplied "
                    "by CLV in the ROI math, so miscalibration would distort every dollar figure."
                )
            else:
                st.info("Run `python src/evaluate.py` to generate the ROC/PR/calibration charts.")