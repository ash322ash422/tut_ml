from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

app = Flask(__name__)

scalar_path = Path.cwd().parent / 'model_saved' / 'scaler.pkl'
model_path = Path.cwd().parent / 'model_saved' / 'rfc.pkl' # other options: rfc.pkl, lr.pkl

scaler=pickle.load(open(scalar_path,'rb'))
model=pickle.load(open(model_path,'rb'))

@app.route('/')
def hello_world():
	return render_template("index.html")

@app.route('/result', methods=['POST'])
def result():

    Age = request.form.get("Age")
    Age = int(Age)

    Glucose = request.form.get("Glucose")
    Glucose = int(Glucose)

    BloodPressure = request.form.get("BloodPressure")
    BloodPressure = int(BloodPressure)

    Insulin = request.form.get("Insulin")
    Insulin = int(Insulin)

    BMIs = request.form.get("BMI")
    BMIs = float(BMIs)

    SkinThickness = request.form.get("SkinThickness")
    SkinThickness = int(SkinThickness)

    DiabetesPedigreeFunction = request.form.get("DiabetesPedigreeFunction")
    DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)

    temp_arr=list()
    temp_arr=temp_arr+[Glucose, BloodPressure, SkinThickness, Insulin, BMIs, DiabetesPedigreeFunction, Age]

    data=np.array([temp_arr])
    temp_sc=scaler.transform(data)
    pred=model.predict(temp_sc)[0]
    pred=round(pred, 2)
    print(temp_arr)     
    print(temp_sc)   
    print(pred)

    if pred==0:
        res="does not indicate"
    if pred==1:
        res="indicates"

    return render_template('result.html', prediction=res)


if __name__ == '__main__':
	app.run(debug=True)
