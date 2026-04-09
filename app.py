import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------- LOAD FILES --------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("💳 CreditWise Loan Approval System")
st.write("Fill the details below:")

# -------- INPUTS --------
Applicant_Income = st.number_input("Applicant Income", min_value=0)
Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0)
Age = st.slider("Age", 18, 70, 30)
Dependents = st.selectbox("Dependents", [0, 1, 2, 3])
Existing_Loans = st.selectbox("Existing Loans", [0, 1, 2, 3])

Savings = st.number_input("Savings", min_value=0)
Collateral_Value = st.number_input("Collateral Value", min_value=0)
Loan_Amount = st.number_input("Loan Amount", min_value=0)
Loan_Term = st.selectbox("Loan Term (months)", [12, 36, 60, 120, 180, 240, 360])

Education_Level = st.selectbox("Education Level", [0, 1])

Employment_Status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
Marital_Status = st.selectbox("Marital Status", ["Single", "Married"])

Loan_Purpose = st.selectbox("Loan Purpose", ["Car", "Education", "Home", "Personal"])
Property_Area = st.selectbox("Property Area", ["Semiurban", "Urban", "Rural"])

Gender = st.selectbox("Gender", ["Male", "Female"])
Employer_Category = st.selectbox("Employer Category", ["Government", "MNC", "Private", "Unemployed"])

DTI_Ratio = st.slider("DTI Ratio", 0.0, 1.0, 0.3)
Credit_Score = st.slider("Credit Score", 300, 900, 650)

# -------- PREDICTION --------
if st.button("Predict"):

    # create dictionary with all columns = 0
    input_data = dict.fromkeys(columns, 0)

    # -------- NUMERICAL --------
    input_data["Applicant_Income"] = Applicant_Income
    input_data["Coapplicant_Income"] = Coapplicant_Income
    input_data["Age"] = Age
    input_data["Dependents"] = Dependents
    input_data["Existing_Loans"] = Existing_Loans
    input_data["Savings"] = Savings
    input_data["Collateral_Value"] = Collateral_Value
    input_data["Loan_Amount"] = Loan_Amount
    input_data["Loan_Term"] = Loan_Term
    input_data["Education_Level"] = Education_Level

    # -------- ONE HOT ENCODING --------

    # Employment
    if "Employment_Status_Self-employed" in input_data:
        input_data["Employment_Status_Self-employed"] = 1 if Employment_Status == "Self-employed" else 0
    if "Employment_Status_Unemployed" in input_data:
        input_data["Employment_Status_Unemployed"] = 1 if Employment_Status == "Unemployed" else 0

    # Marital
    if "Marital_Status_Single" in input_data:
        input_data["Marital_Status_Single"] = 1 if Marital_Status == "Single" else 0

    # Loan Purpose
    if "Loan_Purpose_Car" in input_data:
        input_data["Loan_Purpose_Car"] = 1 if Loan_Purpose == "Car" else 0
    if "Loan_Purpose_Education" in input_data:
        input_data["Loan_Purpose_Education"] = 1 if Loan_Purpose == "Education" else 0
    if "Loan_Purpose_Home" in input_data:
        input_data["Loan_Purpose_Home"] = 1 if Loan_Purpose == "Home" else 0

    # Property Area
    if "Property_Area_Semiurban" in input_data:
        input_data["Property_Area_Semiurban"] = 1 if Property_Area == "Semiurban" else 0
    if "Property_Area_Urban" in input_data:
        input_data["Property_Area_Urban"] = 1 if Property_Area == "Urban" else 0

    # Gender
    if "Gender_Male" in input_data:
        input_data["Gender_Male"] = 1 if Gender == "Male" else 0

    # Employer Category
    if "Employer_Category_Government" in input_data:
        input_data["Employer_Category_Government"] = 1 if Employer_Category == "Government" else 0
    if "Employer_Category_MNC" in input_data:
        input_data["Employer_Category_MNC"] = 1 if Employer_Category == "MNC" else 0
    if "Employer_Category_Private" in input_data:
        input_data["Employer_Category_Private"] = 1 if Employer_Category == "Private" else 0
    if "Employer_Category_Unemployed" in input_data:
        input_data["Employer_Category_Unemployed"] = 1 if Employer_Category == "Unemployed" else 0

    # -------- FEATURE ENGINEERING --------
    if "DTI_Ratio_sq" in input_data:
        input_data["DTI_Ratio_sq"] = DTI_Ratio ** 2
    if "Credit_Score_sq" in input_data:
        input_data["Credit_Score_sq"] = Credit_Score ** 2

    # -------- CONVERT TO DATAFRAME (IMPORTANT FIX) --------
    final_df = pd.DataFrame([input_data])

    # -------- SCALE --------
    final_data = scaler.transform(final_df)

    # -------- PREDICT --------
    result = model.predict(final_data)

    if result[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")