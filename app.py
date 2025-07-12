import streamlit as st
import pickle
import tensorflow as tf
import pandas as pd

model = tf.keras.models.load_model("model.h5")
ohe = pickle.load(open("ohe.pkl", "rb"))
le = pickle.load(open("le.pkl", "rb"))

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Customer Churn Prediction")
st.markdown("Enter The Details Below to Predict the Customer Churn")

with st.form("input_form"):
    sex = st.selectbox("Sex", ["Male", "Female"])
    senior_citizen = st.selectbox("Is Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure", min_value=1, max_value=100)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_services = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
    payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.slider("Monthly Charges", 0, 200)
    total_charges = st.slider("Total Charges", 0, 200)

    submit = st.form_submit_button("Predict")


if submit:
    cat_input = pd.DataFrame([{
        'gender': sex,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_services,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method
    }])

    ohe_encoder = ohe.transform(cat_input)

    numeric_input = pd.DataFrame([{
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])

    final_df = pd.concat([numeric_input.reset_index(drop=True), pd.DataFrame(ohe_encoder)], axis=1)
    final_df = final_df.astype('float32')
    
    prediction = model.predict(final_df)
    
    churn_prob = prediction[0][0]

    churn_pred = (churn_prob > 0.5).astype(int)

    pred = le.inverse_transform([churn_pred])[0]

    if pred == "Yes":
        st.error(f"High Risk of Churn With Prob: {churn_prob:.2%}")
    
    else:
        st.success(f"Low Risk of Churn With Prob: {churn_prob:.2f}")



