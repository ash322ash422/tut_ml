"""
generate_data.py
----------------
Creates a synthetic borrower dataset and saves it as borrowers.csv.
Run this once before opening the notebook.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

# ── Core financial features ────────────────────────────────────────────────
annual_income       = np.random.randint(150_000, 2_500_000, N)
credit_score        = np.random.randint(300, 851, N)
debt_to_income      = np.round(np.random.uniform(0.05, 0.90, N), 2)
loan_amount         = np.random.randint(50_000, 2_000_000, N)
employment_years    = np.random.randint(0, 31, N)
missed_payments     = np.random.randint(0, 10, N)
num_credit_accounts = np.random.randint(1, 15, N)
collateral_value    = np.random.randint(0, 3_000_000, N)

# ── Non-traditional / alternative data ────────────────────────────────────
mobile_recharge_regularity = np.round(np.random.uniform(0.1, 1.0, N), 2)
upi_txn_per_month          = np.random.randint(0, 60, N)
ecom_prepaid_ratio         = np.round(np.random.uniform(0.0, 1.0, N), 2)
bounced_transactions       = np.random.randint(0, 8, N)
salary_credit_regularity   = np.round(np.random.uniform(0.0, 1.0, N), 2)

# ── Inject missing values (~8%) to simulate real-world dirty data ──────────
def inject_missing(arr, pct=0.08):
    arr = arr.astype(float)
    idx = np.random.choice(len(arr), size=int(pct * len(arr)), replace=False)
    arr[idx] = np.nan
    return arr

collateral_value           = inject_missing(collateral_value)
mobile_recharge_regularity = inject_missing(mobile_recharge_regularity)
ecom_prepaid_ratio         = inject_missing(ecom_prepaid_ratio)
salary_credit_regularity   = inject_missing(salary_credit_regularity)
employment_years           = inject_missing(employment_years)

# ── Build default label using realistic underwriting logic ─────────────────
risk = (
    (credit_score < 580).astype(int) * 2 +
    (debt_to_income > 0.60).astype(int) * 2 +
    (missed_payments >= 3).astype(int) * 2 +
    (annual_income < 400_000).astype(int) +
    (bounced_transactions >= 3).astype(int) +
    (loan_amount > annual_income * 3).astype(int)
)
default = (risk >= 3).astype(int)
# Add 5% random noise so the task is not trivially easy
flip = np.random.choice(N, size=int(0.05 * N), replace=False)
default[flip] = 1 - default[flip]

# ── Assemble DataFrame ─────────────────────────────────────────────────────
df = pd.DataFrame({
    "customer_id":                range(1001, 1001 + N),
    "annual_income":              annual_income,
    "credit_score":               credit_score,
    "debt_to_income":             debt_to_income,
    "loan_amount":                loan_amount,
    "employment_years":           employment_years,
    "missed_payments":            missed_payments,
    "num_credit_accounts":        num_credit_accounts,
    "collateral_value":           collateral_value,
    "mobile_recharge_regularity": mobile_recharge_regularity,
    "upi_txn_per_month":          upi_txn_per_month,
    "ecom_prepaid_ratio":         ecom_prepaid_ratio,
    "bounced_transactions":       bounced_transactions,
    "salary_credit_regularity":   salary_credit_regularity,
    "default":                    default,
})

out = "borrowers.csv"
df.to_csv(out, index=False)
print(f"Saved {out}  |  shape: {df.shape}  |  default rate: {df['default'].mean():.1%}")
print(f"Missing values per column:\n{df.isnull().sum()[df.isnull().sum()>0]}")
