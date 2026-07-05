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
│   └── budget_sensitivity.py  # compares strategies across a range of budget sizes
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
```

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

## Next steps
- Replace assumed effectiveness rates with estimates from a real or simulated
  A/B test comparing "received discount" vs. "no action" outcomes.
- Model diminishing returns — after enough discounts, the assumption that the
  35% effectiveness rate holds uniformly likely breaks down.
- Extend to multi-period simulation: what happens to expected value if the same
  200-customer budget is applied every month for a year?
