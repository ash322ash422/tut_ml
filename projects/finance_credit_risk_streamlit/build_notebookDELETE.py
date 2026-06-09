import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"}
}

def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

cells = []

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
"# 🏦 Credit Risk Modelling — End-to-End Project\n"
"### Data Preparation · Model Development · Interpretation · Deployment\n\n"
"---\n\n"
"This notebook walks through a complete, production-style credit risk pipeline:\n\n"
"| Step | What we do |\n"
"|---|---|\n"
"| **1. Data Preparation** | Load borrower data, explore it, handle missing values |\n"
"| **2. Model Development** | Train Logistic Regression and Random Forest; compare metrics |\n"
"| **3. Interpretation** | Feature importance charts and SHAP explainability plots |\n"
"| **4. Deployment** | Save models with `joblib`; run a live Streamlit scoring app |\n\n"
"> **Dataset:** `data/borrowers.csv` — 1,000 synthetic loan applicants with traditional  \n"
"> financial features (income, credit score) *and* alternative data (mobile, UPI, e-commerce).\n\n"
"> Run `python data/generate_data.py` once if the CSV does not yet exist."
))

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
"---\n"
"## Part 1 — Data Preparation\n\n"
"Good models start with clean data. This section covers:\n"
"- Loading and inspecting the dataset\n"
"- Visualising class balance and feature distributions\n"
"- Handling missing values using scikit-learn imputers\n"
"- Scaling features ready for modelling"
))

# 1a: Imports
cells.append(md("### 1.1 Imports"))
cells.append(code(
"import numpy as np\n"
"import pandas as pd\n"
"import matplotlib.pyplot as plt\n"
"import matplotlib.patches as mpatches\n"
"import seaborn as sns\n"
"import warnings\n"
"warnings.filterwarnings('ignore')\n\n"
"# scikit-learn — preprocessing and models\n"
"from sklearn.model_selection import train_test_split\n"
"from sklearn.preprocessing import StandardScaler\n"
"from sklearn.impute import SimpleImputer\n"
"from sklearn.pipeline import Pipeline\n"
"from sklearn.linear_model import LogisticRegression\n"
"from sklearn.ensemble import RandomForestClassifier\n"
"from sklearn.metrics import (classification_report, confusion_matrix,\n"
"                              ConfusionMatrixDisplay, roc_auc_score,\n"
"                              roc_curve, f1_score)\n\n"
"# Model persistence and explainability\n"
"import joblib\n"
"import shap\n\n"
"plt.rcParams['figure.dpi'] = 110\n"
"sns.set_style('whitegrid')\n"
"np.random.seed(42)\n"
"print('All libraries loaded.')\n"
))

# 1b: Load data
cells.append(md(
"### 1.2 Load & Inspect the Dataset\n\n"
"We load the CSV and immediately check its shape, data types, and how many values are missing."
))
cells.append(code(
"df = pd.read_csv('data/borrowers.csv')\n\n"
"print('Shape:', df.shape)\n"
"print('Default rate: %.1f%%' % (df['default'].mean() * 100))\n"
"print()\n"
"df.head()\n"
))

cells.append(code(
"# Check data types and missing values in one view\n"
"info = pd.DataFrame({\n"
"    'dtype':   df.dtypes,\n"
"    'missing': df.isnull().sum(),\n"
"    'pct_missing': (df.isnull().mean() * 100).round(1)\n"
"})\n"
"print(info[info['missing'] > 0])\n"
"print()\n"
"print('Total missing cells:', df.isnull().sum().sum())\n"
))

# 1c: EDA
cells.append(md(
"### 1.3 Exploratory Data Analysis\n\n"
"Before touching the model, we visualise the data. Two questions we always ask:\n"
"1. Is the target class balanced?\n"
"2. Do the key features look different for defaulters vs non-defaulters?"
))
cells.append(code(
"fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n\n"
"# ── Left: class balance bar chart ────────────────────────────────────────\n"
"counts = df['default'].value_counts()\n"
"bars = axes[0].bar(['No Default (0)', 'Default (1)'], counts.values,\n"
"                   color=['#2196F3', '#F44336'], width=0.5, edgecolor='white')\n"
"for b, v in zip(bars, counts.values):\n"
"    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 5,\n"
"                 '%d\\n(%.0f%%)' % (v, v/len(df)*100),\n"
"                 ha='center', fontsize=10, fontweight='bold')\n"
"axes[0].set_title('Class Balance', fontsize=12, fontweight='bold')\n"
"axes[0].set_ylabel('Count')\n\n"
"# ── Right: credit score KDE split by default ─────────────────────────────\n"
"for cls, color, label in [(0, '#2196F3', 'No Default'), (1, '#F44336', 'Default')]:\n"
"    df[df['default'] == cls]['credit_score'].plot.kde(\n"
"        ax=axes[1], color=color, linewidth=2.5, label=label)\n"
"axes[1].set_title('Credit Score Distribution by Default', fontsize=12, fontweight='bold')\n"
"axes[1].set_xlabel('Credit Score')\n"
"axes[1].legend()\n\n"
"plt.tight_layout()\n"
"plt.show()\n"
))

cells.append(code(
"# ── Correlation heatmap (exclude customer_id and target) ─────────────────\n"
"# Helps spot multicollinearity and which features relate to default\n"
"feat_cols = [c for c in df.columns if c not in ['customer_id', 'default']]\n"
"corr = df[feat_cols + ['default']].corr()\n\n"
"fig, ax = plt.subplots(figsize=(12, 7))\n"
"sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,\n"
"            linewidths=0.5, ax=ax, annot_kws={'size': 8})\n"
"ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')\n"
"plt.tight_layout()\n"
"plt.show()\n"
))

# 1d: Clean
cells.append(md(
"### 1.4 Handle Missing Values & Scale Features\n\n"
"We use a **scikit-learn Pipeline** — this is best practice because:\n"
"- It prevents *data leakage* (imputer/scaler fit only on training data)\n"
"- It packages preprocessing + model into one exportable object\n\n"
"**Strategy:** fill numeric missing values with the column median (robust to outliers)."
))
cells.append(code(
"# ── Define features and target ────────────────────────────────────────────\n"
"FEATURES = [\n"
"    'annual_income', 'credit_score', 'debt_to_income', 'loan_amount',\n"
"    'employment_years', 'missed_payments', 'num_credit_accounts',\n"
"    'collateral_value', 'mobile_recharge_regularity', 'upi_txn_per_month',\n"
"    'ecom_prepaid_ratio', 'bounced_transactions', 'salary_credit_regularity'\n"
"]\n\n"
"X = df[FEATURES]\n"
"y = df['default']\n\n"
"# ── Train / test split — stratified to preserve default rate ──────────────\n"
"X_train, X_test, y_train, y_test = train_test_split(\n"
"    X, y, test_size=0.20, random_state=42, stratify=y\n"
")\n\n"
"print('Train:', X_train.shape, '| default rate: %.1f%%' % (y_train.mean()*100))\n"
"print('Test: ', X_test.shape,  '| default rate: %.1f%%' % (y_test.mean()*100))\n"
"print()\n"
"print('Missing values in X_train:', X_train.isnull().sum().sum())\n"
))

cells.append(code(
"# ── Preprocessing pipeline (impute → scale) ──────────────────────────────\n"
"# SimpleImputer: replaces NaN with the median of that column\n"
"# StandardScaler: centres each feature to mean=0, std=1\n"
"#   (important for Logistic Regression; not needed for Random Forest but harmless)\n"
"preprocessor = Pipeline([\n"
"    ('imputer', SimpleImputer(strategy='median')),\n"
"    ('scaler',  StandardScaler())\n"
"])\n\n"
"# Fit on TRAIN only, then transform both sets\n"
"X_train_proc = preprocessor.fit_transform(X_train)\n"
"X_test_proc  = preprocessor.transform(X_test)      # no fit_transform here!\n\n"
"print('After preprocessing — any NaNs left?')\n"
"print('  Train:', np.isnan(X_train_proc).sum())\n"
"print('  Test: ', np.isnan(X_test_proc).sum())\n"
))

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — MODEL DEVELOPMENT
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
"---\n"
"## Part 2 — Model Development\n\n"
"We train two models often used in credit risk:\n\n"
"| Model | Strengths | Weaknesses |\n"
"|---|---|---|\n"
"| **Logistic Regression** | Simple, fast, highly interpretable, well-understood by regulators | Assumes linear decision boundary |\n"
"| **Random Forest** | Captures non-linear patterns, robust to outliers, high accuracy | Less interpretable out of the box |\n\n"
"We compare them on three metrics:\n"
"- **AUC-ROC** — overall discrimination ability\n"
"- **F1-Score** — balance of precision and recall\n"
"- **Accuracy** — overall correct predictions"
))

# 2a: LR
cells.append(md("### 2.1 Logistic Regression"))
cells.append(code(
"# Logistic Regression estimates P(default) as a sigmoid function of a\n"
"# linear combination of features. Highly interpretable — each coefficient\n"
"# tells us the direction and magnitude of each feature's effect.\n"
"# class_weight='balanced' compensates for class imbalance automatically.\n\n"
"lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)\n"
"lr.fit(X_train_proc, y_train)\n\n"
"lr_pred  = lr.predict(X_test_proc)\n"
"lr_prob  = lr.predict_proba(X_test_proc)[:, 1]   # probability of default\n"
"lr_auc   = roc_auc_score(y_test, lr_prob)\n"
"lr_f1    = f1_score(y_test, lr_pred)\n"
"lr_acc   = lr.score(X_test_proc, y_test)\n\n"
"print('Logistic Regression')\n"
"print('  AUC-ROC:  %.4f' % lr_auc)\n"
"print('  F1 Score: %.4f' % lr_f1)\n"
"print('  Accuracy: %.4f' % lr_acc)\n"
"print()\n"
"print(classification_report(y_test, lr_pred, target_names=['No Default', 'Default']))\n"
))

# 2b: RF
cells.append(md("### 2.2 Random Forest"))
cells.append(code(
"# Random Forest trains hundreds of decision trees on random subsets of data\n"
"# and features, then averages their predictions. This reduces overfitting\n"
"# and typically outperforms a single tree significantly.\n"
"# n_estimators=200: number of trees — more trees = more stable but slower\n\n"
"rf = RandomForestClassifier(\n"
"    n_estimators=200,\n"
"    max_depth=8,             # limits tree depth to prevent overfitting\n"
"    class_weight='balanced', # handles class imbalance\n"
"    random_state=42,\n"
"    n_jobs=-1                # use all CPU cores\n"
")\n"
"rf.fit(X_train_proc, y_train)\n\n"
"rf_pred = rf.predict(X_test_proc)\n"
"rf_prob = rf.predict_proba(X_test_proc)[:, 1]\n"
"rf_auc  = roc_auc_score(y_test, rf_prob)\n"
"rf_f1   = f1_score(y_test, rf_pred)\n"
"rf_acc  = rf.score(X_test_proc, y_test)\n\n"
"print('Random Forest')\n"
"print('  AUC-ROC:  %.4f' % rf_auc)\n"
"print('  F1 Score: %.4f' % rf_f1)\n"
"print('  Accuracy: %.4f' % rf_acc)\n"
"print()\n"
"print(classification_report(y_test, rf_pred, target_names=['No Default', 'Default']))\n"
))

# 2c: Compare
cells.append(md("### 2.3 Model Comparison"))
cells.append(code(
"fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n\n"
"# ── Left & Centre: Confusion Matrices ────────────────────────────────────\n"
"for ax, pred, title in [\n"
"    (axes[0], lr_pred, 'Logistic Regression'),\n"
"    (axes[1], rf_pred, 'Random Forest')\n"
"]:\n"
"    ConfusionMatrixDisplay(\n"
"        confusion_matrix(y_test, pred),\n"
"        display_labels=['No Default', 'Default']\n"
"    ).plot(ax=ax, colorbar=False, cmap='Blues')\n"
"    ax.set_title(title, fontsize=11, fontweight='bold')\n\n"
"# ── Right: ROC Curves ─────────────────────────────────────────────────────\n"
"for prob, auc, label, color in [\n"
"    (lr_prob, lr_auc, 'Logistic Regression', '#E91E63'),\n"
"    (rf_prob, rf_auc, 'Random Forest',       '#2196F3')\n"
"]:\n"
"    fpr, tpr, _ = roc_curve(y_test, prob)\n"
"    axes[2].plot(fpr, tpr, linewidth=2.5, color=color,\n"
"                 label='%s (AUC=%.3f)' % (label, auc))\n\n"
"axes[2].plot([0,1],[0,1],'k--', linewidth=1)\n"
"axes[2].set_xlabel('False Positive Rate')\n"
"axes[2].set_ylabel('True Positive Rate')\n"
"axes[2].set_title('ROC Curve Comparison', fontsize=11, fontweight='bold')\n"
"axes[2].legend(fontsize=9)\n\n"
"plt.suptitle('Model Comparison — Logistic Regression vs Random Forest',\n"
"             fontsize=13, fontweight='bold', y=1.02)\n"
"plt.tight_layout()\n"
"plt.show()\n\n"
"# ── Summary table ─────────────────────────────────────────────────────────\n"
"comparison = pd.DataFrame({\n"
"    'Model':    ['Logistic Regression', 'Random Forest'],\n"
"    'AUC-ROC':  [round(lr_auc,4), round(rf_auc,4)],\n"
"    'F1 Score': [round(lr_f1,4),  round(rf_f1,4)],\n"
"    'Accuracy': [round(lr_acc,4), round(rf_acc,4)]\n"
"})\n"
"print(comparison.to_string(index=False))\n"
))

# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — INTERPRETATION
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
"---\n"
"## Part 3 — Model Interpretation\n\n"
"Regulators and credit committees need to understand *why* a model makes a decision.  \n"
"We use two complementary tools:\n\n"
"- **Feature Importance** (Random Forest built-in) — which features matter most overall\n"
"- **SHAP values** — how each feature pushed a *specific prediction* up or down\n\n"
"> SHAP (SHapley Additive exPlanations) is the gold standard for model explainability.\n"
"> Each SHAP value answers: *'How much did this feature contribute to this borrower's default probability?'*"
))

# 3a: Feature importance
cells.append(md("### 3.1 Random Forest Feature Importance"))
cells.append(code(
"# Feature importance = total impurity reduction caused by each feature\n"
"# across all trees, normalised to sum to 1.0\n\n"
"importance_df = pd.DataFrame({\n"
"    'feature':    FEATURES,\n"
"    'importance': rf.feature_importances_\n"
"}).sort_values('importance', ascending=True)\n\n"
"# Colour: top 5 features in red, rest in blue\n"
"top5 = importance_df.nlargest(5, 'importance')['feature'].tolist()\n"
"colors = ['#F44336' if f in top5 else '#90CAF9' for f in importance_df['feature']]\n\n"
"fig, ax = plt.subplots(figsize=(9, 6))\n"
"bars = ax.barh(importance_df['feature'], importance_df['importance'],\n"
"               color=colors, edgecolor='white')\n"
"for bar, val in zip(bars, importance_df['importance']):\n"
"    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,\n"
"            '%.3f' % val, va='center', fontsize=8)\n"
"ax.set_xlabel('Gini Importance')\n"
"ax.set_title('Random Forest — Feature Importance', fontsize=12, fontweight='bold')\n"
"red_p  = mpatches.Patch(color='#F44336', label='Top 5 features')\n"
"blue_p = mpatches.Patch(color='#90CAF9', label='Other features')\n"
"ax.legend(handles=[red_p, blue_p], fontsize=9)\n"
"plt.tight_layout()\n"
"plt.show()\n"
))

# 3b: LR coefficients
cells.append(md("### 3.2 Logistic Regression Coefficients\n\n"
"For Logistic Regression, the coefficients directly show the direction and strength of each feature's effect.\n"
"- **Positive coefficient** → higher value increases default probability\n"
"- **Negative coefficient** → higher value decreases default probability"))
cells.append(code(
"coef_df = pd.DataFrame({\n"
"    'feature':     FEATURES,\n"
"    'coefficient': lr.coef_[0]\n"
"}).sort_values('coefficient')\n\n"
"colors = ['#F44336' if c > 0 else '#2196F3' for c in coef_df['coefficient']]\n\n"
"fig, ax = plt.subplots(figsize=(9, 6))\n"
"ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors, edgecolor='white')\n"
"ax.axvline(0, color='black', linewidth=1)\n"
"ax.set_xlabel('Coefficient (log-odds scale)')\n"
"ax.set_title('Logistic Regression Coefficients', fontsize=12, fontweight='bold')\n"
"red_p  = mpatches.Patch(color='#F44336', label='Increases default risk')\n"
"blue_p = mpatches.Patch(color='#2196F3', label='Decreases default risk')\n"
"ax.legend(handles=[red_p, blue_p], fontsize=9)\n"
"plt.tight_layout()\n"
"plt.show()\n"
))

# 3c: SHAP
cells.append(md("### 3.3 SHAP Explainability\n\n"
"SHAP gives us two key views:\n"
"1. **Summary Plot (beeswarm)** — overall impact of every feature across all test samples\n"
"2. **Waterfall Plot** — a single borrower's prediction broken down feature by feature"))
cells.append(code(
"# SHAP TreeExplainer is optimised for tree-based models like Random Forest\n"
"# We use a sample of 200 test points to keep computation fast\n\n"
"sample_idx = np.random.choice(len(X_test_proc), size=200, replace=False)\n"
"X_sample   = X_test_proc[sample_idx]\n\n"
"explainer   = shap.TreeExplainer(rf)\n"
"shap_values = explainer.shap_values(X_sample)\n\n"
"# shap_values is a list [class0_values, class1_values]\n"
"# We use index [1] for the Default class\n"
"print('SHAP values computed for', len(X_sample), 'test samples.')\n"
"print('Shape:', np.array(shap_values[1]).shape)\n"
))

cells.append(code(
"# ── SHAP Summary Plot (beeswarm) ─────────────────────────────────────────\n"
"# Each dot = one borrower. Colour = feature value (red=high, blue=low).\n"
"# Position on x-axis = SHAP value (how much it pushed the prediction)\n\n"
"plt.figure(figsize=(10, 7))\n"
"shap.summary_plot(\n"
"    shap_values[1],\n"
"    X_sample,\n"
"    feature_names=FEATURES,\n"
"    show=False\n"
")\n"
"plt.title('SHAP Summary — Impact on Default Probability', fontsize=12, fontweight='bold')\n"
"plt.tight_layout()\n"
"plt.show()\n"
))

cells.append(code(
"# ── SHAP Waterfall Plot — single borrower explanation ────────────────────\n"
"# Pick the test borrower with the highest predicted default probability\n"
"# This is the kind of output a credit officer would see for a flagged application\n\n"
"highest_risk_idx = np.argmax(rf_prob)\n"
"single_shap = explainer(X_test_proc[highest_risk_idx:highest_risk_idx+1])\n\n"
"plt.figure(figsize=(10, 6))\n"
"shap.plots.waterfall(single_shap[0], max_display=13, show=False)\n"
"plt.title('SHAP Waterfall — Why this borrower is high risk', fontsize=11, fontweight='bold')\n"
"plt.tight_layout()\n"
"plt.show()\n\n"
"print('Borrower predicted default probability: %.1f%%' % (rf_prob[highest_risk_idx]*100))\n"
))

# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — DEPLOYMENT
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
"---\n"
"## Part 4 — Deployment Simulation\n\n"
"Once models are trained, we need to:\n"
"1. **Save** them to disk so they can be loaded without retraining\n"
"2. **Load** them in a web app that scores new borrowers in real time\n\n"
"We use `joblib` for serialisation — it handles scikit-learn objects efficiently.\n"
"The Streamlit app (`app.py`) reads these saved files."
))

# 4a: Save
cells.append(md("### 4.1 Save Models with joblib"))
cells.append(code(
"import os\n"
"os.makedirs('models', exist_ok=True)\n\n"
"# Save the preprocessing pipeline — must be saved alongside the model\n"
"# so new data is transformed identically to training data\n"
"joblib.dump(preprocessor,   'models/preprocessor.pkl')\n\n"
"# For the Streamlit app we wrap preprocessor + model into one pipeline\n"
"# so the app only needs to call .predict_proba(raw_features)\n"
"from sklearn.pipeline import Pipeline as SKPipeline\n\n"
"lr_pipeline = SKPipeline([('prep', preprocessor), ('model', lr)])\n"
"rf_pipeline = SKPipeline([('prep', preprocessor), ('model', rf)])\n\n"
"joblib.dump(lr_pipeline, 'models/logistic_regression.pkl')\n"
"joblib.dump(rf_pipeline, 'models/random_forest.pkl')\n\n"
"print('Saved:')\n"
"for f in os.listdir('models'):\n"
"    size = os.path.getsize('models/' + f) / 1024\n"
"    print('  models/%s  (%.0f KB)' % (f, size))\n"
))

# 4b: Reload & verify
cells.append(md("### 4.2 Reload & Verify\n\n"
"Always verify that a freshly loaded model gives identical predictions — a basic sanity check."))
cells.append(code(
"# Load models from disk (simulating what the Streamlit app does)\n"
"lr_loaded = joblib.load('models/logistic_regression.pkl')\n"
"rf_loaded = joblib.load('models/random_forest.pkl')\n\n"
"# Predict on raw (un-preprocessed) test data — the pipeline handles it\n"
"lr_reload_prob = lr_loaded.predict_proba(X_test)[:, 1]\n"
"rf_reload_prob = rf_loaded.predict_proba(X_test)[:, 1]\n\n"
"# Verify predictions are identical to the original (should both be True)\n"
"print('LR predictions match:', np.allclose(lr_reload_prob, lr_prob))\n"
"print('RF predictions match:', np.allclose(rf_reload_prob, rf_prob))\n"
"print()\n"
"print('Models verified. Ready for deployment.')\n"
))

# 4c: Streamlit instructions
cells.append(md("### 4.3 Launch the Streamlit App\n\n"
"The file `app.py` in the project root contains a fully working Streamlit web app.  \n"
"It loads the saved models and lets you score a borrower interactively via sidebar sliders.\n\n"
"**To run it, open a terminal in the project folder and run:**\n\n"
"```bash\n"
"streamlit run app.py\n"
"```\n\n"
"The app will open in your browser at `http://localhost:8501`.\n\n"
"> The app reads feature values from sliders, passes them through the saved pipeline  \n"
"> (imputer → scaler → model), and displays the predicted default probability and risk band."))

cells.append(code(
"# Preview what the app sees: score a single hand-crafted borrower ──────────\n"
"# This is exactly what happens when you move a slider in the Streamlit UI\n\n"
"sample_borrower = pd.DataFrame([{\n"
"    'annual_income':              850_000,\n"
"    'credit_score':               720,\n"
"    'debt_to_income':             0.28,\n"
"    'loan_amount':                300_000,\n"
"    'employment_years':           6,\n"
"    'missed_payments':            0,\n"
"    'num_credit_accounts':        4,\n"
"    'collateral_value':           800_000,\n"
"    'mobile_recharge_regularity': 0.91,\n"
"    'upi_txn_per_month':          18,\n"
"    'ecom_prepaid_ratio':         0.82,\n"
"    'bounced_transactions':       0,\n"
"    'salary_credit_regularity':   0.95\n"
"}])\n\n"
"lr_pd = lr_loaded.predict_proba(sample_borrower)[0][1]\n"
"rf_pd = rf_loaded.predict_proba(sample_borrower)[0][1]\n\n"
"print('Sample Borrower Scoring')\n"
"print('  Logistic Regression PD: %.1f%%' % (lr_pd * 100))\n"
"print('  Random Forest PD:       %.1f%% ' % (rf_pd * 100))\n"
"print('  Ensemble Average PD:    %.1f%%' % ((lr_pd + rf_pd) / 2 * 100))\n"
"print()\n"
"avg = (lr_pd + rf_pd) / 2\n"
"band = 'LOW RISK — Approve' if avg < 0.2 else ('MEDIUM RISK — Review' if avg < 0.5 else 'HIGH RISK — Reject')\n"
"print('  Risk Band:', band)\n"
))

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
"---\n"
"## ✅ Project Summary\n\n"
"| Part | Key tools used | Output |\n"
"|---|---|---|\n"
"| Data Preparation | `pandas`, `SimpleImputer`, `StandardScaler`, `Pipeline` | Clean feature matrix |\n"
"| Model Development | `LogisticRegression`, `RandomForestClassifier` | AUC, F1, ROC curves |\n"
"| Interpretation | `feature_importances_`, `shap.TreeExplainer` | Importance charts, SHAP plots |\n"
"| Deployment | `joblib`, `streamlit` | `.pkl` model files, live scoring app |\n\n"
"### Files in this project\n\n"
"```\n"
"credit_project/\n"
"├── data/\n"
"│   ├── generate_data.py     # run once to create borrowers.csv\n"
"│   └── borrowers.csv        # 1,000 synthetic loan applicants\n"
"├── models/                  # created by the notebook\n"
"│   ├── preprocessor.pkl\n"
"│   ├── logistic_regression.pkl\n"
"│   └── random_forest.pkl\n"
"├── notebooks/\n"
"│   └── credit_risk.ipynb    # this notebook\n"
"├── app.py                   # Streamlit scoring app\n"
"└── requirements.txt         # Python dependencies\n"
"```"
))

# ── Write notebook ────────────────────────────────────────────────────────────
nb.cells = cells
import os
os.makedirs('/home/claude/credit_project/notebooks', exist_ok=True)
out = '/home/claude/credit_project/notebooks/credit_risk.ipynb'
with open(out, 'w') as f:
    nbf.write(nb, f)
print('Written:', out, '| cells:', len(cells))
