import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(layout="wide")

# -----------------------------
# Custom CSS for Background + Overlay
# -----------------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://miro.medium.com/v2/resize:fit:1200/0*DLSc317Usr5g5USD.jpeg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.main-container {
    background: rgba(0, 0, 0, 0.6);
    padding: 40px;
    border-radius: 15px;
    width: 50%;
    margin: auto;
    margin-top: 100px;
    color: white;
}

h1 {
    text-align: center;
    color: white;
}

.stButton>button {
    width: 100%;
    background-color: #ff4b4b;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    height: 45px;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

# -----------------------------
# Overlay Container
# -----------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown("<h1>🏠 Hyderabad House Price Prediction</h1>", unsafe_allow_html=True)

area = st.number_input("Total Area (sq ft)", value=1200)
washrooms = st.number_input("Washrooms", value=2)

bedrooms = st.selectbox(
    "Bedrooms Type",
    ["1 BHK Apartment", "2 BHK Apartment", "3 BHK Apartment", "1 BHK Penthouse"]
)

if st.button("Predict Price"):

    input_dict = {
        'Area': area,
        'Washrooms': washrooms,
        'Bedrooms': bedrooms
    }

    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

    input_scaled = scaler.transform(input_final)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Price: ₹{prediction[0]:,.2f}")

st.markdown('</div>', unsafe_allow_html=True)
