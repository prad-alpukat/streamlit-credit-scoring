import streamlit as st
from datetime import datetime

row1 = st.columns([99,1], gap="small")
column1, column2 = st.columns(2)
column3, column4 = st.columns(2)

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

# Title
with row1[0]:
    st.title("Formulir Input Data Pelanggan")

# Pengelompokan berdasarkan jenis informasi

# 1. Informasi Demografi
with column1:
    st.subheader("Informasi Demografi")
    age = st.number_input("Usia", min_value=18, max_value=100, step=1)
    job = st.selectbox("Pekerjaan", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid',
                                    'management', 'retired', 'self-employed', 'services',
                                    'student', 'technician', 'unemployed', 'unknown'])
    marital = st.selectbox("Status Perkawinan", ['married', 'single', 'divorced', 'unknown'])
    education = st.selectbox("Pendidikan", ['primary', 'secondary', 'tertiary', 'unknown'])

# 2. Informasi Keuangan
with column2:
    st.subheader("Informasi Keuangan")
    balance = st.number_input("Saldo", min_value=0, step=1)
    default = st.selectbox("Kredit Macet", ['yes', 'no'])
    housing = st.selectbox("Pinjaman Rumah", ['yes', 'no'])
    loan = st.selectbox("Pinjaman Pribadi", ['yes', 'no'])

# 3. Informasi Kontak dan Kampanye
with column3:
    st.subheader("group 3.1")
    contact = st.selectbox("Jenis Kontak", ['cellular', 'telephone', 'unknown'])
    day = st.number_input("Hari Terakhir Dihubungi", min_value=1, max_value=31, step=1)
    month = st.selectbox("Bulan Terakhir Dihubungi", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
                                                    'aug', 'sep', 'oct', 'nov', 'dec'])
    duration = st.number_input("Durasi Kontak (detik)", min_value=0, step=1)
with column4:
    st.subheader("group 3.2")
    campaign = st.number_input("Jumlah Kontak dalam Kampanye Saat Ini", min_value=1, step=1)
    pdays = st.number_input("Hari Sejak Kontak Terakhir", min_value=-1, step=1, help="-1 jika tidak pernah dihubungi")
    previous = st.number_input("Jumlah Kontak Sebelumnya", min_value=0, step=1)
    poutcome = st.selectbox("Hasil Kampanye Sebelumnya", ['failure', 'nonexistent', 'success'])

# Tombol untuk submit
if st.button("Submit", type="primary", help="Klik untuk submit data"):
    result = predict_loan_approval(model, age, balance, day, duration, campaign, pdays, previous, job, marital, education, default, housing, loan, contact, month, poutcome)
    st.write("Hasil Prediksi: ", result)
    if(result == 1):
        st.write("Pelanggan dinyatakan layak mendapatkan pinjaman")
    else:   
        st.write("Pelanggan dinyatakan tidak layak mendapatkan pinjaman")