import streamlit as st
import pickle
import pandas as pd

# load model
model = pickle.load(open("pipeline.pkl", "rb"))

st.set_page_config(page_title="Churn Prediction", page_icon="📊")

st.title("📊 Customer Churn Prediction")
st.info("This app predicts whether a customer will churn.")

# sidebar inputs
st.sidebar.header("⚙️ Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure", min_value=0)
monthly = st.sidebar.number_input("Monthly Charges", min_value=0.0)
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox("Payment Method", 
                               ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

if st.button("Predict"):

    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": monthly * tenure
    }])

    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)

    churn_prob = prob[0][1] * 100

    if prediction[0] == 1:
        st.error(f"❌ Customer will CHURN ({churn_prob:.2f}% chance)")
    else:
        st.success(f"✅ Customer will STAY ({100 - churn_prob:.2f}% confidence)")