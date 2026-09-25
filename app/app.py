import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ==============================
# Load saved models and files
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

logistic_model = joblib.load(
    BASE_DIR / "logistic_model.pkl"
)

random_forest_model = joblib.load(
    BASE_DIR / "random_forest_model.pkl"
)

scaler = joblib.load(
    BASE_DIR / "scaler.pkl"
)

feature_columns = joblib.load(
    BASE_DIR / "feature_columns.pkl"
)

# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ==============================
# Title
# ==============================

st.title("📊 Customer Churn Prediction")
st.write(
    "Predict whether a customer is likely to churn based on their profile and service details."
)

st.divider()


# ==============================
# Customer Information
# ==============================

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

with col3:
    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )


# ==============================
# Additional Services
# ==============================

st.subheader("Additional Services")

col1, col2, col3 = st.columns(3)

with col1:
    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes"]
    )

with col3:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes"]
    )


# ==============================
# Billing Information
# ==============================

st.subheader("Billing Information")

col1, col2 = st.columns(2)

with col1:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col2:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )


# ==============================
# Prediction
# ==============================

if st.button("🔮 Predict Customer Churn", type="primary"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [tenure * monthly_charges],
        "gender": [gender],
        "Partner": [partner],
        "Dependents": [dependents],
        "PhoneService": [phone_service],
        "InternetService": [internet_service],
        "Contract": [contract],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method]
    })


    # One-hot encoding
    input_data = pd.get_dummies(
        input_data,
        drop_first=True
    )


    # Match training columns
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # Scale input
    input_scaled = scaler.transform(input_data)


    # Prediction
    prediction = logistic_model.predict(input_scaled)[0]

    probability = logistic_model.predict_proba(
        input_scaled
    )[0][1]


    st.divider()

    # Result
    if prediction == 1:

        st.error("⚠️ Customer is likely to CHURN")

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

        st.warning(
            "This customer may require a retention strategy."
        )

    else:

        st.success("✅ Customer is likely to STAY")

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

        st.info(
            "This customer currently has a lower predicted churn risk."
        )