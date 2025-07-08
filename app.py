import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# ML Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Function: Preprocess data
def preprocess_data(data):
    if 'Dependents' in data.columns:
        data['Dependents'] = data['Dependents'].replace('3+', 3).astype(float)

    data.replace({'Loan_Status': {'N': 0, 'Y': 1}}, inplace=True)
    data.replace({
        'Married': {'No': 0, 'Yes': 1},
        'Gender': {'Male': 1, 'Female': 0},
        'Self_Employed': {'No': 0, 'Yes': 1},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
        'Education': {'Graduate': 1, 'Not Graduate': 0}
    }, inplace=True)
    return data

# Streamlit page setup
st.set_page_config(page_title="Loan Status Predictor", layout="centered")
st.title("🏦 Loan Status Prediction App")
st.write("Upload your dataset and get predictions using different machine learning models.")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your CSV data file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### 🔍 Preview of Data")
    st.write(data.head())

    data = preprocess_data(data)

    # Imputation
    imputer = SimpleImputer(strategy="median")
    X = data.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
    Y = data['Loan_Status']
    X = imputer.fit_transform(X)

    # Train-Test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Model selection
    model_name = st.selectbox("🧠 Select a model", [
        "SVM", "Random Forest", "Logistic Regression", "K-Nearest Neighbors", "Decision Tree"
    ])

    if model_name == "SVM":
        model = SVC(kernel='linear', probability=True)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "K-Nearest Neighbors":
        model = KNeighborsClassifier()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()

    # Train model
    model.fit(X_train, Y_train)
    accuracy = accuracy_score(Y_test, model.predict(X_test))
    st.write(f"### 🎯 Accuracy of {model_name}: **{accuracy:.2f}**")

    # Input form
    st.write("## 📝 Enter Details for Prediction")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=5, value=0)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income (INR)", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income (INR)", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (INR)", min_value=0, value=100)
    loan_term = st.number_input("Loan Amount Term (months)", min_value=0, value=360)
    credit_history = st.selectbox("Credit History", [1, 0])
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

    if st.button("📊 Predict Loan Status"):
        # Prepare user input
        input_df = pd.DataFrame({
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_term],
            'Credit_History': [credit_history],
            'Property_Area': [property_area]
        })

        input_df = preprocess_data(input_df)
        input_df = imputer.transform(input_df)

        prediction = model.predict(input_df)
        confidence = model.predict_proba(input_df)[0][prediction[0]] * 100

        if prediction[0] == 1:
            st.success(f"✅ Your loan is likely to be **Approved**")
        else:
            st.error(f"❌ Your loan is likely to be **Denied**")

        st.write(f"### 🔎 Confidence: **{confidence:.2f}%**")
