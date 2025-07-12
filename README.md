# 📉 Customer Churn Prediction Web App

An interactive web application built using **Streamlit** and powered by a **pre-trained TensorFlow deep learning model**. The app predicts whether a customer is likely to churn based on input features, making it especially useful for telecom and subscription-based services aiming to retain customers.

---

## 🔍 Project Summary

This tool enables businesses to assess customer churn risks by analyzing behavioral and account-related attributes such as service usage, contract details, and payment history. By identifying high-risk customers, businesses can take proactive measures to improve customer retention.

---

## 🧠 Model Details

The prediction model is a deep neural network developed using **TensorFlow Keras**, carefully optimized to avoid overfitting using techniques like regularization, normalization, and dropout.

### 🏗️ Model Structure

```python
Input: 45 features (including numeric and encoded categorical variables)

→ Dense(128, activation='relu', kernel_regularizer=L2)
→ BatchNormalization()
→ Dropout(0.5)

→ Dense(64, activation='relu', kernel_regularizer=L2)
→ BatchNormalization()
→ Dropout(0.5)

→ Dense(32, activation='relu')
→ BatchNormalization()
→ Dropout(0.3)

→ Dense(1, activation='sigmoid')  # Binary output (Churn: Yes/No)


---

## 🖥️ Demo

Try the app on your local machine:

![image](https://github.com/user-attachments/assets/e9e2d47f-c227-482e-adbe-c2ecd75ed369)


---

## 🚀 Getting Started

### 📦 Prerequisites

Make sure you have the following installed:

- Python 3.8+
- `streamlit`
- `tensorflow`
- `pandas`
- `scikit-learn`

Install dependencies using:

```bash
pip install -r requirements.txt
```

### ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📁 Files Included

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit app |
| `model.h5` | Trained TensorFlow model |
| `ohe.pkl` | OneHotEncoder for categorical features |
| `le.pkl` | LabelEncoder for output classes |

---

## 📝 Input Fields

- Gender, Senior Citizen, Partner, Dependents
- Tenure, Phone Service, Internet Service, etc.
- Monthly & Total Charges
- Contract and Payment Method

---

## 📊 Output

- **Prediction Label**: `Yes` or `No`
- **Churn Probability**: Displayed as percentage
- **Risk Message**: High or Low Risk

---

## ✨ Future Improvements

- Upload CSV to predict churn for multiple customers
- Visualize churn patterns with charts
- Deploy on Streamlit Cloud or Hugging Face Spaces
