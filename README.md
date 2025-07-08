# 🏦 LOAN-STATUS-PREDICTION

A **machine learning-based web application** that predicts whether a loan application will be approved or not.  
Built using **Python**, **Scikit-learn**, and **Streamlit**, the app allows you to upload your dataset or manually enter applicant details to get **real-time loan status predictions** with model confidence scores.

---

## 🚀 **Key Features**

- ✅ **Data Preprocessing**
  - Cleans and handles missing values
  - Encodes categorical variables
- ✅ **Multiple ML Models**
  - Support Vector Machine (SVM)
  - Random Forest
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
- ✅ **Performance**
  - Achieves ~83% test accuracy
  - Hyperparameter tuning for better results
  - Evaluates using accuracy, precision, recall, F1 score
- ✅ **Interactive Web App**
  - Upload CSV datasets for batch predictions
  - Use manual input form for single predictions
  - Displays prediction result and model confidence

---

## 🛠️ **Tech Stack**

- **Python 3.x**
- **Pandas**
- **Scikit-learn**
- **Streamlit**

---

## Features
- Upload your CSV data
- Choose between SVM and Random Forest models
- Visualize feature importance
- Get prediction results with personalized tips

## Installation

```
pip install -r requirements.txt
streamlit run app.py
```

Upload a dataset that includes columns like: Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, etc.
