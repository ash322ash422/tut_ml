import streamlit as st
import joblib
import numpy as np
from pathlib import Path
import os

def load_model():
    """
    Load the trained model based on the environment (local or Streamlit Cloud).
    """
    model = 'model.pkl'
    on_streamlit_share = os.getenv("STREAMLIT_SERVER_RUN_ON_SAVE")
    if on_streamlit_share is None:  # running on local machine
        model_path = Path.cwd() / model
    else:  # running on Streamlit Cloud
        model_path = Path.cwd() / 'tutorial' / 'tut_ml' / 'bigmart_sales_prediction' / model # 
            
    return joblib.load(model_path)

def get_input_features():
    """
    Get input features from the user via Streamlit interface.
    """
    numerical_features = ['Item_Weight', 'Item_MRP']
    input_features = []

    # Collect numerical inputs
    for feature_name in numerical_features:
        value = st.number_input(f"Enter value for {feature_name}", step=0.01)
        input_features.append(value)

    # Define mappings for categorical features
    feature_texts = [
        ('Item_Fat_Content', {'Low Fat': 0, 'Regular': 1}),
        ('Item_Type', {
            'Fruits and Vegetables': 0, 'Snack Foods': 1, 'Household': 2, 'Frozen Foods': 3,
            'Dairy': 4, 'Canned': 5, 'Baking Goods': 6, 'Health and Hygiene': 7,
            'Soft Drinks': 8, 'Meat': 9, 'Breads': 10, 'Hard Drinks': 11, 'Others': 12,
            'Starchy Foods': 13, 'Breakfast': 14, 'Seafood': 15
        }),
        ('Outlet_Size', {'Medium': 0, 'Small': 1, 'High': 2}),
        ('Outlet_Location_Type', {'Tier 3': 0, 'Tier 2': 1, 'Tier 1': 2}),
        ('Outlet_Type', {
            'Supermarket Type1': 0, 'Grocery Store': 1, 'Supermarket Type3': 2, 'Supermarket Type2': 3
        })
    ]

    # Collect categorical inputs
    for feature_name, feature_dict in feature_texts:
        value = st.selectbox(f"Select value for {feature_name}:", list(feature_dict.keys()))
        input_features.append(feature_dict[value])

    return input_features

def predict_sales(model, input_features):
    """
    Make predictions using the trained model.
    """
    input_features_array = np.array(input_features).reshape(1, -1)
    predicted_sales = model.predict(input_features_array)
    return predicted_sales[0]

def main():
    """
    Main function to run the Streamlit app.
    """
    trained_model = load_model()
    st.title('Sales Prediction')
    input_features = get_input_features()

    if st.button('Predict'):
        predicted_sales = predict_sales(trained_model, input_features)
        st.write(f"Predicted sales: {predicted_sales:.2f}")

if __name__ == "__main__":
    main()
