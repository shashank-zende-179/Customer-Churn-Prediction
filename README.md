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

