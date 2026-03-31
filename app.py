import streamlit as st
import pickle
import pandas as pd

# page config
st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="centered")

# load model
model = pickle.load(open("pipeline.pkl", "rb"))

# sidebar inputs
st.sidebar.title("⚙️ Input Parameters")

tenure = st.sidebar.number_input("📅 Tenure (months)", min_value=0)
monthly = st.sidebar.number_input("💰 Monthly Charges", min_value=0.0)
contract = st.sidebar.selectbox("📄 Contract Type", 
                                ["Month-to-month", "One year", "Two year"])

# main UI
st.title("📊 Customer Churn Prediction")
st.markdown("### Predict whether a customer will leave or stay")

st.divider()

# predict button
if st.button("🔍 Predict", use_container_width=True):

    # FULL input (fix for missing columns)
    input_df = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": monthly,
        "TotalCharges": monthly * tenure
    }])

    # prediction
    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)

    churn_prob = prob[0][1] * 100

    st.divider()

    # result
    if prediction[0] == 1:
        st.error(f"❌ Customer will CHURN ({churn_prob:.2f}% chance)")
    else:
        st.success(f"✅ Customer will STAY ({100 - churn_prob:.2f}% confidence)")

    # chart
    st.subheader("📊 Prediction Confidence")

    chart_data = pd.DataFrame({
        "Result": ["Stay", "Churn"],
        "Probability": [1 - prob[0][1], prob[0][1]]
    })

    st.bar_chart(chart_data.set_index("Result"))

# footer
st.markdown("---")
st.caption("Built by Chandu 🚀")