"""
genai_agent.py

Two GenAI-powered features on top of the churn model:

1. Q&A AGENT — ask natural-language questions about your results
   ("why is churn high among month-to-month customers?") and get answers
   grounded in your ACTUAL data, not the model's guesses. This uses real
   tool-calling (function calling): the model decides which whitelisted
   aggregation to run against your dataframe, the code executes it, and
   the model answers using the real numbers that come back. This is the
   core "agent" pattern — plan, call a tool, use the result, respond.

2. OUTREACH MESSAGE GENERATOR — takes one customer's profile, their SHAP
   explanation (why the model flagged them), and the recommended action,
   and drafts a short, personalized retention email a rep could actually
   send.

Uses Groq's API (free tier — no billing required; fast inference on
open models like Llama 3.3, with OpenAI-compatible tool calling).

SETUP REQUIRED:
   1. Get a free API key: https://console.groq.com/keys
   2. Set it as an environment variable before running:
       Windows (PowerShell):  $env:GROQ_API_KEY = "gsk_..."
       Mac/Linux:              export GROQ_API_KEY="gsk_..."

Run:
    python src/genai_agent.py
"""

import os
import json
import pandas as pd
from groq import Groq

MODEL_NAME = "llama-3.3-70b-versatile"  # free tier on Groq, supports tool calling

DATA_PATH = "data/featured_telco.csv"

# Whitelisted columns/metrics for the Q&A agent's tool. Restricting to a
# known-safe set of operations (rather than letting the model write and
# execute arbitrary code) avoids the security risk of running LLM-written
# code against your data, while still letting it answer real questions.
GROUPABLE_COLUMNS = ["Contract", "PaymentMethod", "tenure_bucket", "InternetService"]


def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable not set. "
            "Get a free key from https://console.groq.com/keys and set it before running this script."
        )
    return Groq(api_key=api_key)


# --- Tool implementation: the actual pandas logic the agent can call ---
def query_dataframe(df: pd.DataFrame, group_by: str, metric: str) -> dict:
    if group_by not in GROUPABLE_COLUMNS:
        return {"error": f"'{group_by}' is not a supported grouping column. "
                          f"Choose from: {GROUPABLE_COLUMNS}"}

    if metric == "churn_rate":
        if "Churn" in df.columns:
            result = df.groupby(group_by)["Churn"].apply(lambda x: (x == "Yes").mean())
        elif "churn_prob" in df.columns:
            result = df.groupby(group_by)["churn_prob"].mean()
        else:
            return {"error": "No churn column found in data."}
    elif metric == "avg_clv":
        if "CLV" not in df.columns:
            return {"error": "No CLV column found in data."}
        result = df.groupby(group_by)["CLV"].mean()
    elif metric == "count":
        result = df.groupby(group_by).size()
    else:
        return {"error": f"Unsupported metric '{metric}'. Choose from: churn_rate, avg_clv, count"}

    return {"group_by": group_by, "metric": metric, "results": result.round(4).to_dict()}


def calculate_max_discount(clv: float, success_probability: float, campaign_cost: float) -> dict:
    """
    Computes the maximum discount (in dollars) that can be offered to a
    customer without the retention action becoming a net loss.

    Formula: max_discount = (CLV * success_probability) - campaign_cost

    This exists specifically so the agent NEVER free-generates this number.
    Business-critical math like "how much can we afford to discount" must
    come from a deterministic calculation, not a language model's guess —
    an LLM asked this directly will often invent plausible-sounding but
    wrong numbers with no grounding in the actual CLV or cost figures.
    """
    max_discount = (clv * success_probability) - campaign_cost
    return {
        "clv": clv,
        "success_probability": success_probability,
        "campaign_cost": campaign_cost,
        "expected_value_if_saved": round(clv * success_probability, 2),
        "max_discount": round(max_discount, 2),
        "formula": "max_discount = (CLV * success_probability) - campaign_cost",
        "note": (
            "Any discount at or below this amount keeps the expected value "
            "of the retention action non-negative. Above this amount, the "
            "expected cost of the discount exceeds the expected value of "
            "retaining the customer."
        ),
    }


def calculate_action_roi(clv: float, success_probability: float, action_cost: float) -> dict:
    """
    Computes expected value and ROI for a single retention action.

    expected_value = (clv * success_probability) - action_cost
    roi = expected_value / action_cost   (return per dollar spent)

    Same rationale as calculate_max_discount: exact numeric answers about
    cost-effectiveness must be computed, never generated freehand.
    """
    expected_value = (clv * success_probability) - action_cost
    roi = (expected_value / action_cost) if action_cost > 0 else None
    return {
        "clv": clv,
        "success_probability": success_probability,
        "action_cost": action_cost,
        "expected_value": round(expected_value, 2),
        "roi": round(roi, 3) if roi is not None else None,
        "formula": "expected_value = (CLV * success_probability) - action_cost; roi = expected_value / action_cost",
    }


def lookup_customer_row(df: pd.DataFrame, row_index: int, budget: int = None) -> dict:
    """
    Looks up a specific customer by row index and reports their churn
    probability, CLV, expected value, recommended action, and — crucially —
    whether they fall within the CURRENT retention budget, and their exact
    rank if not.

    This exists so the agent can answer "why wasn't row X selected?" or
    "what about row X?" for ANY row in the dataset, not just the one
    currently selected in the UI — instead of defaulting to "I don't know."
    """
    if row_index not in df.index:
        valid_range = f"{df.index.min()}-{df.index.max()}"
        return {"error": f"Row {row_index} does not exist in the current dataset (valid range: {valid_range})."}

    row = df.loc[row_index]
    result = {"row_index": row_index}

    if "churn_prob" in df.columns:
        result["churn_prob"] = round(float(row["churn_prob"]), 4)
    if "CLV" in df.columns:
        result["CLV"] = round(float(row["CLV"]), 2)
    if "expected_value" in df.columns:
        result["expected_value"] = round(float(row["expected_value"]), 2)
        # Rank among all customers by expected value (1 = highest)
        rank = int((df["expected_value"] > row["expected_value"]).sum() + 1)
        result["rank_by_expected_value"] = rank
        result["total_customers"] = len(df)
        if budget is not None:
            result["current_budget"] = budget
            result["within_current_budget"] = rank <= budget
            if rank > budget:
                result["reason_not_selected"] = (
                    f"Ranked #{rank} by expected value out of {len(df)} customers, "
                    f"which falls below the current budget cutoff of {budget}."
                )
    if "recommended_action" in df.columns:
        result["recommended_action"] = row["recommended_action"]

    return result


# Groq uses the OpenAI-compatible tool schema format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_dataframe",
            "description": (
                "Get aggregated statistics from the customer dataset, grouped by a "
                "column. Use this to answer any question about patterns in the data "
                "(e.g. churn rate by contract type, average CLV by payment method)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                        "enum": GROUPABLE_COLUMNS,
                        "description": "Which column to group customers by.",
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["churn_rate", "avg_clv", "count"],
                        "description": "Which metric to compute per group.",
                    },
                },
                "required": ["group_by", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_max_discount",
            "description": (
                "Calculates the exact maximum discount (in dollars) that can be "
                "offered to a customer without the retention action resulting in a "
                "net loss. ALWAYS use this tool for any question about maximum "
                "discount, break-even discount, or 'how much can we afford to "
                "offer' — never estimate or guess this number directly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "clv": {"type": "number", "description": "Customer lifetime value in dollars."},
                    "success_probability": {
                        "type": "number",
                        "description": "Probability the retention action successfully prevents churn (0 to 1).",
                    },
                    "campaign_cost": {
                        "type": "number",
                        "description": "Fixed cost of running the retention campaign, in dollars (not counting the discount itself).",
                    },
                },
                "required": ["clv", "success_probability", "campaign_cost"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_action_roi",
            "description": (
                "Calculates exact expected value and ROI for a retention action "
                "given its cost and success probability. ALWAYS use this tool for "
                "any question comparing retention actions, computing ROI, or "
                "asking whether an action is worth the cost — never estimate "
                "these numbers directly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "clv": {"type": "number", "description": "Customer lifetime value in dollars."},
                    "success_probability": {
                        "type": "number",
                        "description": "Probability the action successfully prevents churn (0 to 1).",
                    },
                    "action_cost": {"type": "number", "description": "Cost of the action in dollars."},
                },
                "required": ["clv", "success_probability", "action_cost"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_customer_row",
            "description": (
                "Look up a SPECIFIC customer by their row number/index, returning "
                "their churn probability, CLV, expected value, recommended action, "
                "and whether they fall within the current retention budget (and "
                "their exact rank if not). ALWAYS use this tool when asked about "
                "a specific row or customer by number — including rows OTHER than "
                "the currently selected one. Never say 'I don't know' about a row "
                "number without calling this tool first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "row_index": {
                        "type": "integer",
                        "description": "The row number/index of the customer to look up.",
                    },
                },
                "required": ["row_index"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a data analyst assistant for a customer retention system.

CRITICAL RULE: for ANY question involving a specific dollar amount, percentage,
discount, ROI, or break-even calculation, you MUST call the appropriate tool
(calculate_max_discount, calculate_action_roi, or query_dataframe). NEVER
compute or estimate these numbers yourself in your head — always use a tool,
even if you think you know the answer. If the question doesn't give you
enough information to call a tool (e.g. no CLV or cost figures were
provided), ask the user for the missing numbers rather than guessing typical
values.

If asked about a specific customer or row number — even one that is NOT the
currently selected customer — ALWAYS call lookup_customer_row for that row
number. Never respond with "I don't know" about a row number without calling
this tool first. If the row wasn't selected in the current target list,
explain why using the rank and budget information the tool returns (e.g.
"Row 5127 wasn't targeted because it ranked #312 by expected value, below
the current budget cutoff of 200").

When you do have a tool result, explain it in plain language, but the
number itself must always come from the tool's output, never from your own
estimation."""


def ask_question(question: str, df: pd.DataFrame, client: Groq = None,
                  app_context: str = None, budget: int = None) -> str:
    """
    Agentic Q&A loop: send the question + tool definitions to the model,
    execute any tool calls it makes (against the real dataframe, or via
    exact calculation functions for business math), feed results back, and
    return the final grounded answer. A system prompt forces the model to
    use the calculation tools for any numeric business question instead of
    generating numbers freehand — see calculate_max_discount's docstring
    for why this matters.

    app_context: optional string describing the CURRENT STATE of the app
    (e.g. "The user currently has Row 5127 selected: 74% churn probability,
    $10,645 CLV, expected value $4,200, recommended action: email, top SHAP
    factor: MonthlyCharges"). When provided, this is appended to the system
    prompt so the agent can answer questions like "why was this customer
    selected?" or "why email and not a discount?" using the app's actual
    current state, instead of only being able to answer generic aggregate
    questions about the whole dataset.

    budget: the current retention budget (number of customers being
    targeted), passed through to lookup_customer_row so the agent can
    explain why a given row was or wasn't included in the current plan.
    """
    client = client or get_client()

    system_content = SYSTEM_PROMPT
    if app_context:
        system_content += (
            "\n\nCURRENT APP STATE (use this to answer questions about "
            "'this customer', 'the selected customer', 'why was this one "
            "chosen', etc. — these numbers are already computed, don't "
            "recompute them unless asked to check the math):\n" + app_context
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, tools=TOOLS, tool_choice="auto",
    )

    # Agent loop: keep executing tool calls until the model gives a final text answer
    max_turns = 5
    for _ in range(max_turns):
        message = response.choices[0].message
        if not message.tool_calls:
            return message.content  # final text answer

        messages.append(message)

        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "query_dataframe":
                result = query_dataframe(df, **args)
            elif tool_call.function.name == "calculate_max_discount":
                result = calculate_max_discount(**args)
            elif tool_call.function.name == "calculate_action_roi":
                result = calculate_action_roi(**args)
            elif tool_call.function.name == "lookup_customer_row":
                result = lookup_customer_row(df, budget=budget, **args)
            else:
                result = {"error": f"Unknown tool: {tool_call.function.name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=TOOLS, tool_choice="auto",
        )

    return response.choices[0].message.content


def generate_outreach_message(
    customer: dict,
    top_factors: list,
    recommended_action: str,
    client: Groq = None,
) -> str:
    """
    Drafts a short, personalized retention email using the customer's
    profile, the model's SHAP explanation (why they're at risk), and the
    recommended retention action. `top_factors` should be a list of
    (feature_name, direction) tuples like [("MonthlyCharges", "increases risk")].
    """
    client = client or get_client()

    factors_text = "; ".join(f"{name} ({direction})" for name, direction in top_factors)

    prompt = f"""You are a customer retention specialist drafting a short, warm,
non-pushy email to a customer who is at risk of churning.

Customer profile:
- Tenure: {customer.get('tenure')} months
- Monthly charges: ${customer.get('MonthlyCharges', 0):.2f}
- Contract type: {customer.get('Contract')}

Model's key reasons for flagging this customer as at-risk: {factors_text}

Recommended retention action: {recommended_action}

Write a short email (under 150 words) that:
- Sounds genuinely appreciative of their business, not scripted
- Does NOT explicitly mention "churn," "at risk," or that they were flagged by a model
- Naturally incorporates the recommended action ({recommended_action}) as an offer
- Has a warm, human tone — not corporate boilerplate

Return only the email body, no subject line, no preamble."""

    response = client.chat.completions.create(
        model=MODEL_NAME, messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main():
    df = pd.read_csv(DATA_PATH)
    client = get_client()

    print("=== Q&A Agent Demo ===")
    demo_questions = [
        "Which contract type has the highest churn rate?",
        "What's the average CLV for customers who pay by electronic check?",
    ]
    for q in demo_questions:
        print(f"\nQ: {q}")
        answer = ask_question(q, df, client)
        print(f"A: {answer}")

    print("\n\n=== Outreach Message Generator Demo ===")
    example_customer = df.iloc[0].to_dict()
    example_factors = [
        ("MonthlyCharges", "increases risk"),
        ("Contract_Month-to-month", "increases risk"),
        ("tenure", "decreases risk"),
    ]
    message = generate_outreach_message(
        example_customer, example_factors, "10% discount for 3 months", client
    )
    print(message)


if __name__ == "__main__":
    main()