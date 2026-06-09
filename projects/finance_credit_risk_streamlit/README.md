# 🏦 Credit Risk Modelling — End-to-End Project

A teaching project covering the full credit risk ML pipeline:
Data Preparation → Model Development → Interpretation → Deployment.

## Project Structure

```
credit_project/
├── models/                  ← created by the notebook
│   ├── preprocessor.pkl
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
├── notebooks/
│   └── credit_risk.ipynb    ← main teaching notebook
│   ├── generate_data.py     ← run this first
│   └── borrowers.csv        ← generated dataset (1,000 borrowers)
├── app.py                   ← Streamlit scoring app
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
python notebooks/generate_data.py
```

### 3. Open the notebook
```bash
jupyter notebook notebooks/credit_risk.ipynb
```
Run all cells top to bottom. This trains and saves the models.

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

## What Each Part Covers

| Part | Topics |
|---|---|
| **1 — Data Preparation** | Load CSV, EDA, correlation heatmap, missing value imputation, feature scaling, Pipeline |
| **2 — Model Development** | Logistic Regression, Random Forest, AUC-ROC, F1, confusion matrix, ROC curve comparison |
| **3 — Interpretation** | Feature importance, LR coefficients, SHAP beeswarm plot, SHAP waterfall plot |
| **4 — Deployment** | joblib save/load, pipeline serialisation, Streamlit live scoring app |

## Dataset Features

| Feature | Type | Description |
|---|---|---|
| annual_income | int | Borrower annual income (₹) |
| credit_score | int | Credit bureau score (300–850) |
| debt_to_income | float | Total debt / gross income ratio |
| loan_amount | int | Requested loan amount (₹) |
| employment_years | float | Years at current employer |
| missed_payments | int | Late payments in last 24 months |
| num_credit_accounts | int | Open credit lines |
| collateral_value | float | Pledged asset value (₹) |
| mobile_recharge_regularity | float | Recharge consistency score (0–1) |
| upi_txn_per_month | int | Monthly UPI transactions |
| ecom_prepaid_ratio | float | Prepaid e-commerce order fraction |
| bounced_transactions | int | Failed payments |
| salary_credit_regularity | float | Salary credit consistency (0–1) |
| **default** | int | **Target: 1 = defaulted, 0 = repaid** |
