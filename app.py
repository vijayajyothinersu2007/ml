import streamlit as st
import pandas as pd
import joblib

# Load everything
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl") # Load the saved columns

st.title("🏠 Hyderabad House Price Prediction")

# Inputs (adjust these to match your actual CSV column names)
area = st.number_input("Total Area (sq ft)", value=1200)
washrooms = st.number_input("Washrooms", value=2)
# Using text for bedrooms since your model expects categorical strings
bedrooms = st.selectbox("Bedrooms Type", ["1 BHK Apartment", "2 BHK Apartment", "3 BHK Apartment", "1 BHK Penthouse"]) 

if st.button("Predict Price"):
    # 1. Create a dictionary of the input
    input_dict = {
        'Area': area,
        'Washrooms': washrooms,
        'Bedrooms': bedrooms
    }
    
    # 2. Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # 3. Apply One-Hot Encoding
    input_encoded = pd.get_dummies(input_df)
    
    # 4. CRITICAL FIX: Align columns with the model's training data
    # This adds the missing 'Bedrooms_...' columns and fills them with 0
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # 5. Scale and Predict
    input_scaled = scaler.transform(input_final)
    prediction = model.predict(input_scaled)
    
    st.success(f"Predicted Price: ₹{prediction[0]:,.2f}")