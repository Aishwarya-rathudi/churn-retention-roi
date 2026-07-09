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
    }
]


def ask_question(question: str, df: pd.DataFrame, client: Groq = None) -> str:
    """
    Agentic Q&A loop: send the question + tool definition to the model,
    execute any tool calls it makes against the real dataframe, feed
    results back, and return the final grounded answer.
    """
    client = client or get_client()

    messages = [{"role": "user", "content": question}]

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
            if tool_call.function.name == "query_dataframe":
                args = json.loads(tool_call.function.arguments)
                result = query_dataframe(df, **args)
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