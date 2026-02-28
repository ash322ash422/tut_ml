import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ----------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Health_Insurance.xlsm", sheet_name="Health_Insurance")
    return df

df = load_data()

# ----------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------
df["sex"] = df["sex"].map({"male": 1, "female": 0})
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
df = pd.get_dummies(df, columns=["region"], drop_first=True)

X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------------------
# 3. Train Models
# ----------------------------------------------------
models = {}

# Linear Regression
lr = LinearRegression().fit(X_train, y_train)
models["Linear Regression"] = lr

# Decision Tree
dt = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
models["Decision Tree"] = dt

# Random Forest (tuned)
rf = RandomForestRegressor(random_state=42)
param_rf = {"n_estimators": [100, 200], "max_depth": [None, 5, 10]}
rand_rf = RandomizedSearchCV(rf, param_rf, n_iter=3, cv=3, scoring="r2", random_state=42, n_jobs=-1)
rand_rf.fit(X_train, y_train)
models["Random Forest"] = rand_rf.best_estimator_

# Gradient Boosting (tuned)
gb = GradientBoostingRegressor(random_state=42)
param_gb = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 4]}
rand_gb = RandomizedSearchCV(gb, param_gb, n_iter=3, cv=3, scoring="r2", random_state=42, n_jobs=-1)
rand_gb.fit(X_train, y_train)
models["Gradient Boosting"] = rand_gb.best_estimator_

# ----------------------------------------------------
# 4. Pick Best Model
# ----------------------------------------------------
results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append((name, r2, rmse))

best_model_name, best_r2, best_rmse = max(results, key=lambda x: x[1])
best_model = models[best_model_name]

# Save best model + scaler
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# ----------------------------------------------------
# 5. Streamlit UI
# ----------------------------------------------------
st.set_page_config(page_title="Insurance Charge Predictor", page_icon="💰")
st.title("💰 Health Insurance Charges Predictor")
st.write(f"✅ Best Model Used: **{best_model_name}** (R² = {best_r2:.3f})")

# User input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Preprocess input
input_data = pd.DataFrame({
    "age": [age],
    "sex": [1 if sex == "male" else 0],
    "bmi": [bmi],
    "children": [children],
    "smoker": [1 if smoker == "yes" else 0],
    "region": [region]
})

input_data = pd.get_dummies(input_data, columns=["region"], drop_first=True)
expected_cols = X.columns
input_data = input_data.reindex(columns=expected_cols, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Insurance Charge"):
    prediction = best_model.predict(input_scaled)[0]
    st.success(f"💵 Estimated Insurance Charge: ${prediction:,.2f}")
