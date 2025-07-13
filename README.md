# 💳 Credit Card Fraud Detection System

A machine learning project to detect fraudulent credit card transactions using a Random Forest model, SMOTE oversampling, and Streamlit for interactive dashboards.

## 📂 Project Structure
- `preprocess_and_train.py`: Loads data, applies SMOTE, trains model
- `app.py`: Streamlit dashboard for live prediction
- `fraud_model.pkl`: Saved trained model (locally used)
- `creditcard.csv`: Input dataset (from Kaggle)
- `requirements.txt`: Python dependencies

## 🚀 Live Demo
👉 https://credit-card-fraud-detection-cctkrqdxczusa6ksraveoj.streamlit.app/

## 💡 Features
- SMOTE balancing for imbalanced data
- RandomForest-based fraud classification
- Real-time prediction using a Streamlit dashboard
   https://raw.githubusercontent.com/Prasanna-555/Credit-card-fraud-detection/refs/heads/main/Screenshot%201.png


## 📦 How to Run Locally

```bash
pip install -r requirements.txt
python preprocess_and_train.py
streamlit run app.py
