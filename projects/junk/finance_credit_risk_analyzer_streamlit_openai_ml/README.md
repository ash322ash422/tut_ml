# 🏦 Credit Risk Analyzer

A teaching project demonstrating how to combine a **Scikit-Learn ML model** with the **OpenAI API** inside a **Streamlit** web app — all deployable to Streamlit Cloud in minutes.

---

## Architecture

```
User (Streamlit sliders)
        │
        ▼
   RandomForestClassifier  ──►  approval_prob  ──►  Decision Banner
        │
        ▼
   GPT-4o-mini  ──────────────────────────────►  Formal Bank Letter
```

1. **User** adjusts income, loan amount, credit history, etc. via sliders
2. **ML Model** outputs an objective approval probability
3. **OpenAI** writes a formal approval / adverse-action letter based on the score

---

## Project Structure

```
credit_risk_analyzer/
├── app.py              ← Streamlit frontend
├── train_model.py      ← Trains & saves the RandomForest model
├── credit_model.pkl    ← Pre-trained model (committed to repo)
├── sample_data.csv     ← Synthetic training data
├── notebook.ipynb      ← Step-by-step teaching notebook
└── requirements.txt    ← Python dependencies
```

All files live in one flat directory — no sub-packages needed.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Retrain the model
python train_model.py

# 3. Run the app
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Deploy to Streamlit Cloud

1. Push this directory to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

> The `credit_model.pkl` file **must** be committed to the repo so Streamlit Cloud can load it at runtime.

---

## Teaching Notes

| File | What to teach |
|------|--------------|
| `notebook.ipynb` | Full walkthrough: data → EDA → training → evaluation → OpenAI integration |
| `train_model.py` | How to generate synthetic data and save a model with `joblib` |
| `app.py` | Streamlit layout, `@st.cache_resource`, OpenAI client usage |

The notebook covers 7 sections with beginner / intermediate / advanced exercises at the end.

---

## Regulatory Context

The OpenAI prompt instructs the model to write letters compliant with:
- **ECOA** (Equal Credit Opportunity Act)
- **FCRA** (Fair Credit Reporting Act)

This is a simplified demo — not legal advice.
