# Customer Churn Prediction (Machine Learning)

## 📌 Project Overview
This project predicts customer churn using machine learning techniques.  
The goal is to identify whether a customer is likely to leave the service based on demographic and usage-related features.

⚠️ Note: This project uses a **synthetic/demo dataset**, which results in **100% accuracy** and is intended for learning and demonstration purposes.

---

## 📊 Dataset
- File: `Telco-Customer-Churn.csv`
- Records: 500 customers
- Features include:
  - Gender
  - SeniorCitizen
  - Tenure
  - MonthlyCharges
  - TotalCharges
  - Contract type
  - Payment method
  - Churn (Target variable)

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn

---

## 🔄 Project Workflow
1. Data loading and inspection
2. Data cleaning and preprocessing
3. Categorical feature encoding
4. Train-test split
5. Model training:
   - Logistic Regression
   - Random Forest
6. Model evaluation using:
   - Accuracy score
   - Confusion matrix
   - Classification report

---

## 📈 Model Performance
### Logistic Regression
- Accuracy: **100%**

### Random Forest
- Accuracy: **100%**

> High accuracy is due to the synthetic nature of the dataset.

---

## ▶️ How to Run the Project
```bash
python churn_model.py
# Customer Churn Prediction

Complete end-to-end project (data + code) generated and ready to run.

## Run
pip install pandas scikit-learn
python customer_churn_prediction.py
