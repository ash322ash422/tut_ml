"""
app.py  —  Credit Risk Prediction App
--------------------------------------
A simple Streamlit demo that loads the trained models and lets you
score a single borrower in real time.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config 
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="centered"
)

# ── Load models (must run the notebook first to generate these files) ─────────
MODEL_DIR = "models"

@st.cache_resource
def load_models():
    """Load pre-trained models from disk. Cached so they load only once."""
    lr  = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    rf  = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    return lr, rf

# ── Header 
st.title("🏦 Credit Risk Prediction")
st.markdown(
    "Enter a borrower's details below. The app scores them using two trained models "
    "— **Logistic Regression** and **Random Forest** — and shows the estimated "
    "probability of default (PD)."
)
st.divider()

# ── Check models exist 
if not os.path.exists(os.path.join(MODEL_DIR, "random_forest.pkl")):
    st.error(
        "⚠️  Model files not found. Please run the Jupyter notebook first "
        "to train and save the models, then restart this app."
    )
    st.stop()

lr_model, rf_model = load_models()

# ── Sidebar: borrower input form 
st.sidebar.header("📋 Borrower Details")

annual_income      = st.sidebar.number_input("Annual Income (₹)", 50_000, 5_000_000, 600_000, step=10_000)
credit_score       = st.sidebar.slider("Credit Score", 300, 850, 680)
debt_to_income     = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.35, 0.01)
loan_amount        = st.sidebar.number_input("Loan Amount (₹)", 10_000, 5_000_000, 400_000, step=10_000)
employment_years   = st.sidebar.slider("Employment Years", 0, 30, 5)
missed_payments    = st.sidebar.slider("Missed Payments (last 24m)", 0, 10, 0)
num_credit_accounts = st.sidebar.slider("Number of Credit Accounts", 1, 15, 3)
collateral_value   = st.sidebar.number_input("Collateral Value (₹)", 0, 5_000_000, 500_000, step=10_000)

st.sidebar.subheader("📱 Alternative Data")
mobile_recharge    = st.sidebar.slider("Mobile Recharge Regularity", 0.0, 1.0, 0.85, 0.01)
upi_txn            = st.sidebar.slider("UPI Transactions / Month", 0, 60, 15)
ecom_prepaid       = st.sidebar.slider("E-Com Prepaid Ratio", 0.0, 1.0, 0.70, 0.01)
bounced_txn        = st.sidebar.slider("Bounced Transactions", 0, 8, 0)
salary_regularity  = st.sidebar.slider("Salary Credit Regularity", 0.0, 1.0, 0.92, 0.01)

# ── Assemble feature vector (same column order as training) 
feature_names = [
    "annual_income", "credit_score", "debt_to_income", "loan_amount",
    "employment_years", "missed_payments", "num_credit_accounts",
    "collateral_value", "mobile_recharge_regularity", "upi_txn_per_month",
    "ecom_prepaid_ratio", "bounced_transactions", "salary_credit_regularity"
]

input_data = pd.DataFrame([[
    annual_income, credit_score, debt_to_income, loan_amount,
    employment_years, missed_payments, num_credit_accounts,
    collateral_value, mobile_recharge, upi_txn, ecom_prepaid,
    bounced_txn, salary_regularity
]], columns=feature_names)

# ── Predict 
st.subheader("📊 Model Predictions")

col1, col2 = st.columns(2)

lr_prob  = lr_model.predict_proba(input_data)[0][1]
rf_prob  = rf_model.predict_proba(input_data)[0][1]
lr_label = "❌ Default" if lr_prob >= 0.5 else "✅ No Default"
rf_label = "❌ Default" if rf_prob >= 0.5 else "✅ No Default"

with col1:
    st.metric("Logistic Regression", lr_label, f"PD = {lr_prob:.1%}")
    st.progress(float(lr_prob))

with col2:
    st.metric("Random Forest", rf_label, f"PD = {rf_prob:.1%}")
    st.progress(float(rf_prob))

# ── Risk band 
avg_pd = (lr_prob + rf_prob) / 2
st.divider()
st.subheader("🎯 Risk Band (Ensemble Average)")

if avg_pd < 0.20:
    st.success(f"**LOW RISK** — Average PD: {avg_pd:.1%}  |  Recommendation: Approve")
elif avg_pd < 0.50:
    st.warning(f"**MEDIUM RISK** — Average PD: {avg_pd:.1%}  |  Recommendation: Review")
else:
    st.error(f"**HIGH RISK** — Average PD: {avg_pd:.1%}  |  Recommendation: Reject")

# ── Input summary table 
st.divider()
with st.expander("🔍 View input feature values"):
    st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)

st.caption("Models trained on synthetic data for educational purposes only.")
