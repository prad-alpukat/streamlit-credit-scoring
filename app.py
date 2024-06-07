import streamlit as st
import numpy as np
import joblib

def custom_label_encode(arr):
    unique_vals = np.unique(arr)
    encode_dict = {val: idx for idx, val in enumerate(unique_vals)}
    return np.vectorize(encode_dict.get)(arr), encode_dict

def custom_standard_scale(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    return (arr - mean) / std if std != 0 else arr

def predict_loan_approval(model, age, balance, day, duration, campaign, pdays, previous, 
                          job, marital, education, default, housing, loan, contact, month, poutcome):
    # Create a dictionary to store the data
    data_dict = {
        'age': np.array([age]),
        'balance': np.array([balance]),
        'day': np.array([day]),
        'duration': np.array([duration]),
        'campaign': np.array([campaign]),
        'pdays': np.array([pdays]),
        'previous': np.array([previous]),
        'job': np.array([job]),
        'marital': np.array([marital]),
        'education': np.array([education]),
        'default': np.array([default]),
        'housing': np.array([housing]),
        'loan': np.array([loan]),
        'contact': np.array([contact]),
        'month': np.array([month]),
        'poutcome': np.array([poutcome])
    }
    
    # List of numerical columns to scale
    numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    
    # Custom encode categorical features
    encode_dicts = {}
    for feature in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']:
        data_dict[feature], encode_dicts[feature] = custom_label_encode(data_dict[feature])
    
    # Custom scale numerical features
    for feature in numerical_columns:
        data_dict[feature] = custom_standard_scale(data_dict[feature])

    # Combine all features into a single NumPy array
    data_array = np.column_stack([data_dict[feature] for feature in data_dict.keys()])

    # Predict the loan approval
    prediction = model.predict(data_array)
    return prediction[0].tolist()

model = joblib.load("model.pkl")

st.title("Loan Approval Prediction")
st.write("This is a simple loan approval prediction app. Please enter the required information in the sidebar and click the 'Predict' button to get the prediction.")

result = predict_loan_approval( model,58,2143,5,261,1,-1,0,'management','married','tertiary','no','yes','no','unknown','may','unknown')
st.write(result)

# input
age = st.slider("Age", 18, 100, 18)
balance = st.slider("Balance", -10000, 100000, -10000)
day = st.slider("Day", 1, 31, 1)
duration = st.slider("Duration", 0, 5000, 0)
campaign = st.slider("Campaign", 0, 60, 0)
pdays = st.slider("Pdays", -1, 871, -1)
previous = st.slider("Previous", 0, 275, 0)
job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"])
marital = st.selectbox("Marital", ["divorced", "married", "single"])
education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
default = st.selectbox("Default", ["no", "yes"])
housing = st.selectbox("Housing", ["no", "yes"])
loan = st.selectbox("Loan", ["no", "yes"])
contact = st.selectbox("Contact", ["cellular", "telephone", "unknown"])
month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
poutcome = st.selectbox("Poutcome", ["failure", "other", "success", "unknown"])

if st.button("Predict"):
    result = predict_loan_approval(model, age, balance, day, duration, campaign, pdays, previous, job, marital, education, default, housing, loan, contact, month, poutcome)
    st.write(result)

    