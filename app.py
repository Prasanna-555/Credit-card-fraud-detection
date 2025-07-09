import streamlit as st
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

model = load_model()

st.set_page_config(page_title="💳 Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection Dashboard")
st.write("Upload a credit card transaction file to detect fraudulent activities.")

uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if 'Class' in data.columns:
            data = data.drop(columns=['Class'])
        if 'Time' in data.columns:
            data = data.drop(columns=['Time'])

        st.success("✅ File uploaded successfully!")
        st.write("Preview of uploaded data:", data.head())

        preds = model.predict(data)
        data['Prediction'] = preds
        fraud_count = (preds == 1).sum()
        non_fraud_count = (preds == 0).sum()

        st.subheader("📊 Prediction Summary")
        st.markdown(f"🟢 **Non-Fraud**: {non_fraud_count} <br> 🔴 **Fraud**: {fraud_count}", unsafe_allow_html=True)

        st.subheader("📄 Sample Predictions")
        st.dataframe(data.head(10))

        st.bar_chart(data['Prediction'].value_counts())

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
else:
    st.info("📌 Please upload a CSV file to start.")
