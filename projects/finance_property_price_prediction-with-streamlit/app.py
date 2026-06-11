import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sys

st.title("House/Property Price Prediction")

st.image("https://loxfordacademy.com/wp-content/uploads/2026/06/house_pic.png", caption="House",width=250)
# st.image("house_pic.png", caption="House",width=250)


# input features
# Let the user pick BHK or RK. If RK is chosen, only allow 1 (default).
BHK_OR_RK = st.selectbox("Select Property Type:", ["BHK", "RK"])  # line ~9
if BHK_OR_RK == "RK":
       # For RK properties, BHK/RK number should be 1 by default and only option
       BHK_number = st.selectbox("Enter BHK or RK (e.g., 2 for 2 BHK):", [1], index=0)
else:
       # For BHK properties, allow selection from 1 to 6
       BHK_number = st.selectbox("Enter BHK or RK (e.g., 2 for 2 BHK):", [1, 2, 3, 4, 5, 6], index=0)
SQUARE_FT = st.number_input("Enter Square Feet Area:", min_value=100, max_value=10000, value=1000)
LOCATION = st.selectbox("Select Location:", ['Bangalore', 'Mysore', 'Ghaziabad', 'Kolkata', 'Kochi', 'Jaipur',
       'Mohali', 'Chennai', 'Siliguri', 'Noida', 'Raigad', 'Bhubaneswar',
       'Wardha', 'Pune', 'Mumbai', 'Nagpur', 'Other', 'Bhiwadi',
       'Faridabad', 'Lalitpur', 'Maharashtra', 'Vadodara',
       'Visakhapatnam', 'Vapi', 'Mangalore', 'Aurangabad', 'Vijayawada',
       'Belgaum', 'Bhopal', 'Lucknow', 'Kanpur', 'Gandhinagar',
       'Pondicherry', 'Agra', 'Ranchi', 'Gurgaon', 'Udupi', 'Indore',
       'Jodhpur', 'Coimbatore', 'Valsad', 'Palghar', 'Surat', 'Varanasi',
       'Amravati', 'Anand', 'Tirupati', 'Secunderabad', 'Raipur',
       'Vizianagaram', 'Thrissur', 'Madurai', 'Chandigarh', 'Shimla',
       'Gwalior', 'Rajkot', 'Sonipat', 'Allahabad', 'Berhampur',
       'Dharuhera', 'Latur', 'Durgapur', 'Panchkula', 'Solapur', 'Durg',
       'Goa', 'Jamshedpur', 'Hazaribagh', 'Jabalpur', 'Morbi', 'Hubli',
       'Karnal', 'Patna', 'Bilaspur', 'Ratnagiri', 'Meerut', 'Jalandhar',
       'Amritsar', 'Ludhiana', 'Alwar', 'Kota', 'Panaji', 'Kolhapur',
       'Ernakulam', 'Bhavnagar', 'Bharuch', 'Asansol', 'Jhansi', 'Margao',
       'Anantapur', 'Eluru', 'Bhilai', 'Dehradun', 'Guntur', 'Jalgaon',
       'Udaipur', 'Neemrana', 'Sindhudurg', 'Kottayam', 'Dhanbad',
       'Navsari', 'Bahadurgarh', 'Nellore', 'Tirunelveli', 'Cuttack',
       'Haridwar', 'Nainital', 'Jamnagar', 'Kanchipuram', 'Karad',
       'Muzaffarpur', 'Gandhidham', 'Junagadh', 'Moradabad', 'Ahmednagar',
       'Palakkad', 'Kannur', 'Karjat', 'Akola', 'Gaya', 'Ajmer',
       'Dharwad', 'Kollam', 'Palwal', 'Aligarh', 'Rudrapur', 'Tenali',
       'Ongole', 'Puri', 'Solan', 'Kakinada', 'Haldwani', 'Bardhaman',
       'Chandrapur', 'Bokaro', 'Bhimavaram', 'Ujjain', 'Mathura',
       'Rewari', 'Shirdi', 'Rohtak', 'Bareilly', 'Gulbarga', 'Jammu',
       'Raigarh'])
latitude = st.number_input("Enter Latitude:", value=0.0)
longitude = st.number_input("Enter Longitude:", value=0.0)
posted_by = st.radio("Posted By:", ["Owner", "Builder", "Dealer"])
Under_construction = st.radio("Is the property Under Construction?", ["Yes", "No"])
READY_TO_MOVE = st.radio("Is the property Ready to Move?", ["Yes", "No"])
RESALE = st.radio("Is the property for Resale?", ["Yes", "No"])
RERA = st.radio("Is the property RERA registered?", ["Yes", "No"])

# Load the trained model
model = joblib.load("model_lgb.pkl")
le_a = joblib.load("location_encoder.pkl")
le_2 = joblib.load("posted_by_encoder.pkl")

if st.button("Predict Price"):
    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'BHK_OR_RK': [0 if BHK_OR_RK == "BHK" else 1],
        'BHK_number': [BHK_number],
        'SQUARE_FT': [SQUARE_FT],
        'LOCATION': [le_a.transform([LOCATION])[0]],
        'latitude': [latitude],
        'longitude': [longitude],
        'posted_by': [le_2.transform([posted_by])[0]],
        'Under_construction': [1 if Under_construction == "Yes" else 0],
        'READY_TO_MOVE': [1 if READY_TO_MOVE == "No" else 0],
        'RESALE': [1 if RESALE == "No" else 0],
        'RERA': [1 if RERA == "No" else 0]
    })

    # Make prediction
    predicted_price = model.predict(input_data)


    st.success(f"The predicted price of the property is: ₹{np.expm1(predicted_price[0]):,.2f} lakhs")
