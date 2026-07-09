# Churn Prediction & Retention ROI Optimizer

Predict which customers are about to churn, then rank them by **expected revenue saved**
if you take a retention action — not just by churn probability.

## Why this project is different
Most churn projects stop at "predict churn, report accuracy." This one adds a
business-value layer: for every customer, we estimate what it's worth to intervene,
and rank accordingly. That's the part hiring managers actually care about.

## Project structure
```
churn-roi/
├── data/                  # raw + processed data (not committed)
├── src/
│   ├── enrich.py              # adds CLV + retention cost columns to raw data
│   ├── features.py            # feature engineering
│   ├── train.py               # model training + comparison
│   ├── simulate.py            # intervention value simulation (budget-constrained)
│   ├── budget_sensitivity.py  # compares strategies across a range of budget sizes
│   ├── explain.py             # SHAP feature importance + per-customer explanations
│   ├── uplift.py              # causal uplift modeling (T-learner) + Qini evaluation
│   └── genai_agent.py         # Q&A agent (tool-calling) + outreach message generator
├── app/
│   └── streamlit_app.py   # deployable demo app
├── notebooks/
│   └── 01_eda.ipynb       # exploratory analysis
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1: Get the data
Download the **Telco Customer Churn** dataset from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Save it as `data/raw_telco.csv`.

(Kaggle requires login, so this step is manual — everything after this is automated.)

## Step 2: Run the pipeline
```bash
python src/enrich.py              # adds CLV + retention cost columns
python src/features.py            # feature engineering
python src/train.py               # trains + compares models
python src/simulate.py            # runs intervention value simulation
python src/budget_sensitivity.py  # compares strategies across budget sizes, saves a chart
python src/explain.py             # SHAP feature importance + a per-customer explanation
python src/uplift.py              # causal uplift modeling (T-learner) + Qini curve
```

## Step 2b: GenAI agent setup (optional, free)
The Q&A agent and outreach message generator use Groq's API, which is
free (no billing required) and runs open models like Llama 3.3 with very
fast inference.

1. Get a free key from [console.groq.com/keys](https://console.groq.com/keys)
   (just needs an account, no credit card)
2. Set it as an environment variable:
   ```powershell
   # Windows PowerShell
   $env:GROQ_API_KEY = "gsk_..."
   ```
   ```bash
   # Mac/Linux
   export GROQ_API_KEY="gsk_..."
   ```
3. Run the demo:
   ```bash
   python src/genai_agent.py
   ```
   This asks two example questions grounded in your actual data (via real
   tool-calling, not the model guessing), then drafts an example retention
   email using a customer's profile and SHAP explanation.

**Note:** the free tier has a per-minute rate limit — plenty for demoing
this project, but if you hit a rate limit error, just wait a few seconds
and retry.

## Step 3: Run the app
```bash
streamlit run app/streamlit_app.py
```

## Step 4: Deploy
Push to GitHub, then deploy free on [Streamlit Community Cloud](https://streamlit.io/cloud)
or [Hugging Face Spaces](https://huggingface.co/spaces). Takes about 10 minutes.

## Results

Using a fixed retention budget of **200 customers** (a realistic constraint — a
retention team can't call or discount everyone), the model was evaluated on a
held-out test set of 1,409 customers:

| Strategy | Customers targeted | Total expected value |
|---|---|---|
| Naive (rank by churn probability alone) | 200 | $101,247.27 |
| Value-based (rank by expected value: P(churn) × CLV − cost) | 200 | $230,087.57 |

**Value-based targeting captured 127% more expected value than the naive approach**,
using the exact same budget. The two strategies only agreed on **16% of who to
target** — meaning 84% of the time, "most likely to churn" and "most worth saving"
pointed to different customers entirely.

### Why the two strategies diverge
Naive targeting chases whoever is statistically most likely to leave — often
low-tenure, low-spend customers, since new customers churn at the highest rates
in this dataset. But saving a customer who wasn't spending much in the first
place isn't worth much. Value-based targeting instead surfaces customers who are
moderately likely to churn *and* represent high lifetime value — a smaller
probability of success multiplied by a much bigger prize.

### Sensitivity to budget size
The improvement isn't constant — it depends on how constrained the budget is.
Run `python src/budget_sensitivity.py` to see this across a range of budget
sizes (50 to "target everyone"); the advantage of value-based targeting is
largest at small/moderate budgets and shrinks toward zero as the budget grows
large enough to target nearly every customer, since at that point there's no
real prioritization decision left to make.

## Limitations
- **CLV is a simulated proxy** (historical spend + projected future spend,
  discounted for month-to-month contracts) — not sourced from finance systems.
- **Retention action effectiveness (35% for discounts, 15% for outreach) is an
  assumption**, not measured from a real experiment. In a real deployment this
  would come from historical A/B tests of past retention campaigns.
- **No live validation** — this shows the *expected* value of smarter targeting
  under stated assumptions, not a measured causal lift from an actual intervention.

## Uplift modeling: a sharper version of "who's worth targeting"

`simulate.py` assumes every customer responds to a discount at the same
rate (e.g., "35% effectiveness for everyone"). `uplift.py` replaces that
assumption with an individualized estimate: a **T-learner** (two separately
trained models — one on treated customers, one on control) predicts each
customer's specific reduction in churn probability if given a discount.

This surfaces a pattern a fixed-rate assumption can't: **some customers'
predicted uplift is negative** — the model estimates the discount would
make them *more* likely to churn (e.g., customers who weren't price-sensitive
in the first place, for whom a discount offer draws attention to price they
weren't otherwise thinking about). Fixed-effectiveness targeting has no way
to catch this; individualized uplift modeling does.

**Evaluation:** since real ground-truth treatment effects are never
observable per-customer (a customer either got the discount or didn't,
never both), uplift models are evaluated with a **Qini curve** — cumulative
incremental churns avoided when targeting by predicted uplift vs. random
targeting. A curve that bows above the random line means the ranking is
finding real signal, not noise.

**Honest caveat:** this uses a *simulated* treatment/control dataset (since
a real historical A/B test wasn't available). The correlation check in
`uplift.py` output validates that the model recovers the *simulated* ground
truth reasonably well — in a real deployment, the Qini curve on a real
historical experiment would be the actual evidence, not this correlation.

## GenAI agent layer

Two features built on top of the model, using Groq's API (free tier):

- **Q&A agent** (`ask_question` in `genai_agent.py`) — uses real tool-calling:
  the model decides which aggregation to run against the actual customer
  dataframe (via a whitelisted `query_dataframe` function, not arbitrary code
  execution), and answers using the real numbers returned. This keeps
  answers grounded in your actual data rather than the model guessing.
- **Outreach message generator** (`generate_outreach_message`) — takes a
  customer's profile, their top SHAP factors, and the recommended retention
  action, and drafts a short, personalized email a retention rep could
  actually send — closing the loop from "here's a risky customer" to
  "here's what to say to them."
