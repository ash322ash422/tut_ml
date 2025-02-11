# Diabetes Detection

A diabetes detection ML model to accurately predict the likelihood of an individual having diabetes based on various features.

## Package required
 pandas==2.2.3, seaborn==0.13.2, matplotlib==3.10.0, scipy==1.15.1, sckit-learn==1.6.1, flask==3.1.0

## Description

Diabetes detection machine learning project is a system that uses algorithms, statistical models and historical data to predict the likelihood of an individual having diabetes. The goal is to use features like Glucose, age, and Blood Pressure to identify individuals who have diabetes or are at risk of developing it. Machine learning models are trained using this data and can then be used to make predictions on new individuals.


## About Dataset

### Context
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.

### Content
Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age (years)
- Outcome: Class variable (0 or 1)

### Sources
(a) Original owners: National Institute of Diabetes and Digestive and
Kidney Diseases
(b) Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu)
Research Center, RMI Group Leader
Applied Physics Laboratory
The Johns Hopkins University
Johns Hopkins Road
Laurel, MD 20707
(301) 953-6231
(c) Date received: 9 May 1990

## Feature Selection

The process of feature selection in machine learning is used to identify and select the most relevant features from a dataset to improve the performance and efficiency of a machine learning model. It is an important step in the model building process as it can help to reduce overfitting, increase model interpretability, and improve the accuracy of predictions.

### Feature Selection methods use in this projects

#### correlation matrix

A correlation matrix can be used to identify highly correlated features, which can then be removed or consolidated. Highly correlated features can cause a problem in machine learning models as they can introduce multicollinearity, which can lead to unstable and unreliable model estimates.

## Outlier Treatment

Outlier treatment is the process of identifying and handling extreme values or observations that are significantly different from other observations in a dataset. Outliers can have a significant impact on the results of data analysis and modeling, and can skew the mean and standard deviation of a dataset.

### Methods used to identify outliers

- **Visualization:** Outliers can be identified by creating visualizations such as box plots, scatter plots, and histograms to identify observations that fall outside of the typical range.

- **Interquartile range (IQR):** Outliers can be identified by calculating the interquartile range (IQR), which is the difference between the first and third quartile. Values that fall outside of 1.5 * IQR above the third quartile or 1.5 * IQR below the first quartile are considered outliers.

### Methods used to treat outliers

- **Imputing the missing values:** This method can be used to replace outliers with the mean, median, or mode of the dataset.


## Features transformation and scaling

**Feature transformation** is the process of applying mathematical functions to the features in a dataset in order to change their distribution or to extract additional information from the data. Feature transformations are often used to improve the performance of machine learning models by making the features more suitable for the model.

- **Label encoding** is a method of converting categorical variables, represented as text values, into numerical values. It assigns a unique numerical value to each category or level of a categorical feature. This is often used as a preprocessing step before training a machine learning model.

- **Target encoding** is a technique used in machine learning to encode categorical variables. It replaces each category with the average value of the target variable for that category. This can improve the performance of a model by allowing it to better handle categorical variables. It is also known as mean encoding or probability encoding.

**Feature scaling** is the process of normalizing the range of values for each feature in a dataset. This is often done to ensure that all features are on a similar scale and to prevent some features from having a greater impact on the outcome of a machine learning model than others.

- **StandardScaler** is a pre-processing method in machine learning used to standardize a dataset by subtracting the mean and scaling to unit variance. It is commonly used for feature scaling before applying a supervised learning algorithm to a dataset. 

## Model Selection

### Selection of model

- **Logistic regression**

- **Naive Bayes classifier** 

- **K-Nearest Neighbors (KNN)** 

- **A Decision Tree Classifier**

- **Support Vector Classifier (SVC)** 

- **Random Forest Classifier (RFC)** 


### Evaluation of algorithm


- **Confusion Matrix**

A confusion matrix is a table that is used to define the performance of a classification algorithm. Each row of the matrix represents the instances in a predicted class, while each column represents the instances in an actual class (or vice versa). The name stems from the fact that it makes it easy to see if the system is confusing two classes (i.e. commonly mislabeling one as another). The diagonal elements represent the number of points for which the predicted label is equal to the true label, while off-diagonal elements are those that are mislabeled by the classifier. It is a useful tool for understanding the performance of a classification algorithm, including the types of errors that the classifier is making.


## Web App for project

**Flask** is a web framework for building web applications using the Python programming language. It is a micro-framework that provides the basic functionality needed to build web applications, such as routing and request handling, without including a lot of additional features or libraries. This makes it lightweight and easy to use, but also allows for flexibility and customization.


After inserting values for the attribute such as the Glucose, age, and Blood Pressure our flask web app will predict the whether report indicates the presence of diabetes or not.


## Installation

### Installation of Diabetes Detection App

To install a Flask project, you can use the following commands in your command line interface:


Install Flask and any other required dependencies by running:
```
pip install flask
```

Create a new file named `app.py` or `main.py` in the project directory:
Copy code
```
touch app.py
```
In the `app.py` or `main.py` file, import Flask, create an instance of the Flask class, and define the routes and functions that will handle the requests.

Run the application by executing:

```
flask run
```
or

```
python app.py
```
or
```
python main.py
```

This will start the development server and make the application accessible at "http://localhost:5000"

If you want to deploy your application to a production environment, you can use a web server like Gunicorn or uWSGI to serve the application.
It's worth noting that you can also use frameworks like Flask-CLI or Flask-Script to create more advanced command-line scripts for your application.

## Authors

- [Ashwani](https://github.com/ash322ash422) (Project Lead) 
  
