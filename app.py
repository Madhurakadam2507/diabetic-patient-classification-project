import streamlit as st
import pandas as pd
import pickle as pkl

scaler = pkl.load(open('scaler.pkl', 'rb'))
model = pkl.load(open('model.pkl', 'rb'))

st.title("Diabetic Patient Prediction Project")

gender = st.selectbox("Select Gender", ['Female', 'Male', 'Other'])
age = st.number_input("Enter Age", min_value = 0, max_value = 100, value = 50)
hypertension = st.selectbox("Select hypertension", ["Yes", "No"])
heart_disease = st.selectbox("Select heart_disease", ["Yes", "No"])
smoking_history = st.selectbox("Select smoking", ['never', 'No Info', 'former', 'not current', 'ever', 'current'])
bmi = st.number_input("Enter BMI", min_value = 20, max_value = 50, value = 28)
HbA1c_level = st.number_input("Enter H1AC Level", min_value = 5.0, max_value = 10.0, value = 6.6, step=0.1)
blood_glucose_level = st.number_input("Enter Blood Glucose Level", min_value = 25, max_value = 500, value = 200)

if gender == 'Female':
    gender = 0
elif gender == 'Male':
    gender = 1
else:
    gender = 2

if smoking_history == 'never':
    smoking_history = 0
elif smoking_history == 'No Info':
    smoking_history = 1
elif smoking_history == 'former' or smoking_history == 'not current':
    smoking_history = 2
elif smoking_history == 'even':
    smoking_history = 3
else:
    smoking_history = 4

if hypertension == "Yes":
    hypertension = 1
else:
    hypertension = 0

if heart_disease == "Yes":
    heart_disease = 1
else:
    heart_disease = 0

if st.button("Predict"):
    myinput = [[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]]
    columns = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    data = pd.DataFrame(data = myinput, columns = columns)
    data_scaled = scaler.transform(data)
    result = model.predict(data_scaled)
    if result[0] == 1:
        st.error("Person in diabetic")
    else:
        st.success("Person in not diabetic")
    