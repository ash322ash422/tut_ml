import streamlit as st
import numpy as np
import joblib

# Load Mobile Price Prediction Model and Scaler
model = joblib.load('mobile_price_prediction_model.pkl')
scaler = joblib.load('scaler_X.pkl')

st.title("Mobile Price Prediction App")

# Input fields for mobile features
Price = st.number_input("Price (USD)", 500, 4500, 2357)
weight = st.number_input("Weight (grams)",0.0,800.0,135.0)
resolution = st.number_input("Resolution (pixels)",0.0,15.00,5.2)
ppi = st.number_input("PPI (pixels per inch)", 100, 900, 424)
cpu_core = st.selectbox("CPU Cores", [0,1,2,4,6,8,16])
cpu_freq = st.number_input("CPU Frequency (GHz)", 0.0,5.0,1.35)
internal_mem = st.selectbox("Internal Memory (GB)",[0,4,8,16,32,64,128,256,512])
ram = st.number_input("RAM (GB)", 1.00, 16.00, 3.00)
rear_cam = st.number_input("Rear Camera (MP)", 2, 108, 13)
front_cam = st.number_input("Front Camera (MP)", 0.0, 50.0, 8.0)
battery = st.number_input("Battery Capacity (mAh)", 1000, 10000, 2610)
thickness = st.number_input("Thickness (mm)", 0.0, 20.0, 7.4)


mobile_input = np.array([[Price,weight, resolution, ppi, cpu_core, cpu_freq, internal_mem, ram, rear_cam, front_cam, battery, thickness]])

if st.button("🔮 Predict Sale"):
    scaled = scaler.transform(mobile_input) 
    prediction = model.predict(scaled)
    st.success(f"📈 Predicted Sale: {int(prediction[0])} USD")


