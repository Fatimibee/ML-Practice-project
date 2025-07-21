import streamlit as st
import joblib
import pandas as pd

#Load pre-trained model
model=joblib.load(r"D:\Desktop\ML PROJECTS\Internship_Fraud_detection\Internship_model1.pkl")

st.title("🎯Internship Fraud Detection🎯")

st.subheader("Predicting Internship is Scam or Genuine")

st.write("Enter the details below to check if the internship is fraud or genuine:")

Title=st.text_input("Enter Title Of Internship")
Description=st.text_input("Enter Description of Internship")

if st.button("Predict"):
    if Title and Description:

        Input_data=pd.DataFrame({
            "Title":[Title],
            "Description":[Description]
        })

        prediction=model.predict(Input_data)

        if prediction==1:
            st.error(" This Internship is a Scam ❌")
        else:
            st.success(" This Internship is Genuine ✅")
    
    else:
        st.warning("Please fill in all fields to make a prediction.")