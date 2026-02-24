## Mobile Price Prediction using Machine Learning
## 📌 Project Overview

This project aims to predict the price range of mobile phones based on their technical specifications using Machine Learning algorithms.
The dataset is cleaned and preprocessed through missing value handling, skewness correction, and outlier treatment, followed by training and evaluating multiple ML models to identify the best-performing model for accurate price prediction.

## 🎯 Objectives

Clean and preprocess raw mobile phone data

Handle missing values effectively

Treat skewed distributions and outliers

Train and evaluate multiple ML models

Compare model performance using accuracy metrics

Predict mobile price range for new data

## 🛠️ Technologies Used

Python

Pandas

NumPy

Matplotlib / Seaborn

Scikit-learn

## 🧹 Data Preprocessing & Cleaning
## 1️⃣ Handling Missing Values

Numerical features filled using mean / median

Categorical features handled using mode encoding

Ensured no null values remain before modeling

## 2️⃣ Skewness Treatment

Identified skewed numerical features

Applied:

Log transformation

Square root transformation

Improved data normality and model performance

## 3️⃣ Outlier Treatment

Outliers detected using:

IQR (Interquartile Range)

Boxplots

Outliers handled via:

Capping (Winsorization)

Removal (where necessary)

## 4️⃣ Feature Scaling

Applied StandardScaler / MinMaxScaler

Ensured features are on a similar scale for better model performance

## 🔍 Exploratory Data Analysis (EDA)

Feature distribution analysis

Correlation heatmap

Relationship between mobile features and price range

Visualization of outliers and skewed data

## 🤖 Machine Learning Models Used

The following models were trained and compared:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

K-Nearest Neighbors (KNN)

Support Vector Regressor (SVR)

## 📊 Model Evaluation

Models were evaluated using:

R² Score

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Performance comparison was done to select the best-performing model for predictions.

## 🔮 Price Prediction

The final model predicts mobile price range based on specifications such as:

RAM

Battery capacity

Camera features

Internal storage

Processor speed

## 🚀 How to Run the Project

## Clone the repository

git clone https://github.com/your-username/Mobile-Price-Prediction.git

## Install required libraries

pip install -r requirements.txt


Run the notebook or script to train models and make predictions
