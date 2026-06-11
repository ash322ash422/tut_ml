"""
app.py  –  Credit Risk Analyzer
================================
A teaching demo that wires together:
  • Scikit-Learn (RandomForestClassifier) for objective credit scoring
  • OpenAI GPT-4o-mini for human-readable approval / adverse-action letters

Run locally:
    streamlit run app.py

Deploy to Streamlit Cloud:
    Push repo to GitHub → connect on share.streamlit.io
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background: #0f1b2d; }
    [data-testid="stSidebar"] * { color: #e8edf3 !important; }
    [data-testid="stSidebar"] .stSlider > label { font-size: 0.82rem; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }

    /* Decision banners */
    .approved-banner {
        background: linear-gradient(135deg, #16a34a, #15803d);
        color: white; border-radius: 12px;
        padding: 1.5rem 2rem; text-align: center;
        font-size: 1.4rem; font-weight: 700; margin-bottom: 1.5rem;
    }
    .denied-banner {
        background: linear-gradient(135deg, #dc2626, #b91c1c);
        color: white; border-radius: 12px;
        padding: 1.5rem 2rem; text-align: center;
        font-size: 1.4rem; font-weight: 700; margin-bottom: 1.5rem;
    }
    .letter-box {
        background: #fffdf5;
        border-left: 4px solid #ca8a04;
        border-radius: 8px;
        padding: 1.5rem 2rem;
        font-family: 'Georgia', serif;
        line-height: 1.8;
        white-space: pre-wrap;
        font-size: 0.93rem;
    }
    h1 { color: #0f1b2d; }
</style>
""", unsafe_allow_html=True)

# ── Load ML model (cached) ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # bundle  = joblib.load("credit_model.pkl")
    
    from pathlib import Path
    MODEL_PATH = Path(__file__).resolve().parent / "credit_model.pkl"
    bundle = joblib.load(MODEL_PATH)
    
    return bundle["model"], bundle["features"]

model, FEATURES = load_model()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – inputs
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🏦 Credit Analyzer")
    st.markdown("---")

    # ── OpenAI key ──────────────────────────────────────────────────────────────
    st.subheader("🔑 OpenAI API Key")
    openai_key = st.text_input(
        "Paste your key",
        type="password",
        placeholder="sk-...",
        help="Used only to call GPT-4o-mini for the letter. Never stored.",
    )
    st.markdown("---")

    # ── Applicant sliders ───────────────────────────────────────────────────────
    st.subheader("👤 Applicant Profile")

    annual_income = st.slider(
        "Annual Income ($)",
        min_value=15_000, max_value=250_000,
        value=65_000, step=1_000,
        format="$%d",
    )
    loan_amount = st.slider(
        "Loan Amount Requested ($)",
        min_value=1_000, max_value=80_000,
        value=20_000, step=500,
        format="$%d",
    )
    credit_history_yrs = st.slider(
        "Credit History (years)",
        min_value=0, max_value=25, value=5,
    )
    num_late_payments = st.slider(
        "Late Payments (past 2 yrs)",
        min_value=0, max_value=15, value=2,
    )
    employment_yrs = st.slider(
        "Employment Duration (years)",
        min_value=0, max_value=30, value=4,
    )
    num_open_accounts = st.slider(
        "Open Credit Accounts",
        min_value=1, max_value=12, value=4,
    )

    st.markdown("---")

    # ── Optional free-text narrative ─────────────────────────────────────────
    st.subheader("📝 Loan Narrative (optional)")
    narrative = st.text_area(
        "Paste applicant's statement",
        placeholder="e.g. I am applying for a home improvement loan. I recently got promoted …",
        height=130,
    )

    analyze_btn = st.button("🔍 Analyze Application", use_container_width=True, type="primary")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════
st.title("Credit Risk Analyzer")
st.caption("Teaching demo — Scikit-Learn + OpenAI | adjust sliders and click Analyze")


# ── Run analysis ──────────────────────────────────────────────────────────────
if analyze_btn:
    # ── Derived feature ─────────────────────────────────────────────────────
    debt_to_income = loan_amount / annual_income

    input_dict = {
        "annual_income":      annual_income,
        "loan_amount":        loan_amount,
        "credit_history_yrs": credit_history_yrs,
        "num_late_payments":  num_late_payments,
        "debt_to_income":     round(debt_to_income, 4),
        "employment_yrs":     employment_yrs,
        "num_open_accounts":  num_open_accounts,
    }

    X_input = pd.DataFrame([input_dict])[FEATURES]

    # ── ML inference ────────────────────────────────────────────────────────
    approval_prob  = model.predict_proba(X_input)[0][1]
    decision_label = "APPROVED" if approval_prob >= 0.5 else "DENIED"
    is_approved    = decision_label == "APPROVED"

    # ── Top section: metrics ────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Income", f"${annual_income:,.0f}")
    col2.metric("Loan Amount", f"${loan_amount:,.0f}")
    col3.metric("Debt-to-Income", f"{debt_to_income:.1%}")
    col4.metric("Approval Probability", f"{approval_prob:.1%}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Decision banner ─────────────────────────────────────────────────────
    if is_approved:
        st.markdown('<div class="approved-banner">✅ APPROVED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="denied-banner">❌ DENIED</div>', unsafe_allow_html=True)

    # ── Feature importance bar chart ────────────────────────────────────────
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig_data    = pd.DataFrame({"Feature": importances.index, "Importance": importances.values})

    st.subheader("📊 Model Feature Importances")
    st.bar_chart(fig_data.set_index("Feature"))

    # ── OpenAI letter ────────────────────────────────────────────────────────
    st.subheader("📄 AI-Generated Bank Letter")

    if not openai_key:
        st.warning("Enter your OpenAI API key in the sidebar to generate the letter.")
    else:
        narrative_section = (
            f"\n\nApplicant's own statement:\n\"{narrative.strip()}\""
            if narrative.strip() else ""
        )

        prompt = f"""You are a senior loan officer at a U.S. regulated bank.
Based on the automated credit scoring result below, write a formal bank letter to the applicant.

─── Credit Score Summary ───
Decision       : {decision_label}
Approval Prob. : {approval_prob:.1%}
Annual Income  : ${annual_income:,.0f}
Loan Requested : ${loan_amount:,.0f}
Debt-to-Income : {debt_to_income:.1%}
Credit History : {credit_history_yrs} years
Late Payments  : {num_late_payments}
Employment     : {employment_yrs} years
Open Accounts  : {num_open_accounts}{narrative_section}
────────────────────────────

Instructions:
- If APPROVED  : write a warm, professional approval letter with the amount and next steps.
- If DENIED    : write an adverse-action notice that complies with the Equal Credit
                 Opportunity Act (ECOA) and Fair Credit Reporting Act (FCRA).
                 List the specific reasons for denial based on the data.
- Use formal letter format: date, salutation, body paragraphs, closing, signature block.
- Do NOT invent figures not listed above.
- Keep the letter under 350 words.
"""

        with st.spinner("GPT-4o-mini is drafting the letter …"):
            try:
                client   = OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=600,
                )
                letter = response.choices[0].message.content
                st.markdown(f'<div class="letter-box">{letter}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"OpenAI error: {e}")

    # ── Raw input summary (for teaching / debugging) ─────────────────────────
    with st.expander("🔬 Raw Input Sent to Model"):
        st.json(input_dict)

else:
    # ── Empty state ─────────────────────────────────────────────────────────
    st.info("👈  Adjust the sliders in the sidebar and click **Analyze Application** to begin.")
    
