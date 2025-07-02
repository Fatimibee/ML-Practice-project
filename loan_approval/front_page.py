import joblib
import streamlit as st
import pandas as pd

# Load trained model
model = joblib.load(r'D:\Desktop\ML PROJECTS\loan_approval\loan_approval_prediction.pkl')

# Streamlit App UI
st.title("Loan Approval Prediction")
st.markdown("This app predicts whether a loan will be approved based on user inputs.")
st.sidebar.header("User Input Feature")

def user_input():
    # Collect input
    no_of_dependents = st.sidebar.selectbox("No of Dependents", [0, 1, 2, 3, 4, 5])
    education = st.sidebar.selectbox("Education Level", ["Graduate", "Not Graduate"])
    self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.sidebar.number_input("Income (Annum)", min_value=0, key="income_annum")
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, key="loan_amount")
    loan_term = st.sidebar.selectbox("Loan Term (in months)", [2, 4, 68, 10, 12, 14, 16, 18, 20])
    cibil_score = st.sidebar.number_input("CIBIL Score", min_value=300, max_value=900, step=1, key="cibil")
    residential_assets_value = st.sidebar.number_input("Residential Assets Value", min_value=0, key="residential_assets")
    commercial_assets_value = st.sidebar.number_input("Commercial Assets Value", min_value=0, key="commercial_assets")
    luxury_assets_value = st.sidebar.number_input("Luxury Assets Value", min_value=0, key="luxury")
    bank_asset_value = st.sidebar.number_input("Bank Asset Value", min_value=0, key="bank_assets")  # corrected label

    # Encode categorical features (same as used during training)
    education_encoded = 0 if education == "Graduate" else 1
    self_employed_encoded = 1 if self_employed == "Yes" else 0

    # Create DataFrame in same order as training
    data = pd.DataFrame({
        'no_of_dependents': [no_of_dependents],
        'education': [education_encoded],
        'self_employed': [self_employed_encoded],
        'income_annum': [income_annum],
        'loan_amount': [loan_amount],
        'loan_term': [loan_term],
        'cibil_score': [cibil_score],
        'residential_assets_value': [residential_assets_value],
        'commercial_assets_value': [commercial_assets_value],
        'luxury_assets_value': [luxury_assets_value],
        'bank_asset_value': [bank_asset_value]
    })
    return data

# Get user input as X_test
X_test = user_input()

# Predict when button is clicked
if st.button("Predict loan approval"):
    prediction = model.predict(X_test)
    if prediction[0] == 0:
        st.success("🎉 Congratulations! Your loan is likely to be approved.")
    else:
        st.error("❌ Sorry! Your loan application is likely to be rejected. Please try again later.")
