💰 Health Insurance Charges Prediction

This project predicts medical insurance charges based on patient details such as age, sex, BMI, smoking status, children, and region.
It uses machine learning models to train on historical data and provides an easy-to-use Streamlit UI for predictions.

📂 Project Structure
├── Health_Insurance.xlsm   # Dataset
├── app.py                  # Streamlit app
├── requirements.txt        # Dependencies
├── model.pkl               # Trained best model (auto-saved)
├── scaler.pkl              # Scaler used for preprocessing
└── README.md               # Project description

⚙️ Features
Exploratory Data Analysis (EDA): Feature distributions, correlations, and outlier detection.
Data Preprocessing: Missing value check, encoding categorical variables, feature scaling.
Model Training: Linear Regression, Decision Tree, Random Forest, Gradient Boosting.
Hyperparameter Tuning: RandomizedSearchCV for optimal parameters.
Model Evaluation: Metrics like RMSE, R², Adjusted R² with overfitting check.
Best Model Selection: Automatically picks the best-performing model.
Streamlit Web App: Simple UI where users input details and get predicted charges.

🧪 Models Compared
Model	Train RMSE	Test RMSE	Train R²	Test R²	Overfitting
Linear Regression	...	...	...	...	...
Decision Tree	...	...	...	...	...
Random Forest	...	...	...	...	...
Gradient Boosting	...	...	...	...	...
✅ Best Model	...	...	...	...	...

🚀 How to Run
1.Clone the repository: git clone https://github.com/Kirtiverma64/Medical-Insurance-Cost-Prediction.git
  cd insurance-prediction
2.Install dependencies: pip install -r requirements.txt
3.Run the Streamlit app: streamlit run app.py
Enter details (age, BMI, smoker, etc.) in the UI → Get insurance charge prediction 💵

🛠️ Tech Stack
Python (pandas, numpy, scikit-learn)
Streamlit (UI)
Matplotlib & Seaborn (EDA)
GitHub (version control)

📊 Example UI
<img width="1106" height="534" alt="image" src="https://github.com/user-attachments/assets/a3a26e08-80df-4d57-84d5-385f501d9b00" />
<img width="995" height="523" alt="image" src="https://github.com/user-attachments/assets/6b4adcef-d6d3-48e9-8f1c-4b8db937f858" />



🙌 Acknowledgements
Dataset inspired by Kaggle - Insurance Dataset
Built as part of a Machine Learning + Streamlit practice project

📝 Author
👤 Kirti Verma
