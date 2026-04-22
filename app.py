import streamlit as st
import pandas as pd
import pickle as pkl

# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="Diabetes Predictor 🩺",
    page_icon="🧬",
    layout="wide"
)

# -------- LOAD MODEL --------
scaler = pkl.load(open('scaler.pkl', 'rb'))
model = pkl.load(open('model.pkl', 'rb'))

# -------- HEADER --------
st.markdown("""
    <h1 style='text-align: center; color: #ff4b4b;'>🧬 Diabetes Prediction System</h1>
    <p style='text-align: center; font-size:18px;'>
        Enter patient details below to predict diabetes risk instantly ⚡
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# -------- INPUT SECTION --------
st.subheader("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("👤 Gender", ['Female', 'Male', 'Other'])
    age = st.number_input("🎂 Age", min_value=0, max_value=100, value=50)
    hypertension = st.selectbox("💓 Hypertension", ["Yes", "No"])
    heart_disease = st.selectbox("❤️ Heart Disease", ["Yes", "No"])

with col2:
    smoking_history = st.selectbox(
        "🚬 Smoking History",
        ['never', 'No Info', 'former', 'not current', 'ever', 'current']
    )
    bmi = st.number_input("⚖️ BMI", min_value=20.0, max_value=50.0, value=28.0)
    HbA1c_level = st.number_input("🧪 HbA1c Level", min_value=5.0, max_value=10.0, value=6.6, step=0.1)
    blood_glucose_level = st.number_input("🩸 Blood Glucose Level", min_value=25, max_value=500, value=200)

st.markdown("---")

# -------- DATA PROCESSING --------
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
elif smoking_history in ['former', 'not current']:
    smoking_history = 2
elif smoking_history == 'ever':
    smoking_history = 3
else:
    smoking_history = 4

hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0

# -------- PREDICTION BUTTON --------
if st.button("🔍 Predict Diabetes Risk"):
    
    myinput = [[gender, age, hypertension, heart_disease,
                smoking_history, bmi, HbA1c_level, blood_glucose_level]]
    
    columns = ['gender', 'age', 'hypertension', 'heart_disease',
               'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    
    data = pd.DataFrame(myinput, columns=columns)
    data_scaled = scaler.transform(data)
    result = model.predict(data_scaled)

    st.markdown("### 🧾 Prediction Result")

    if result[0] == 1:
        st.error("⚠️ High Risk: The person is likely **Diabetic**")
        st.progress(90)

        # 📍 NEXT STEPS (ADDED)
        st.markdown("### 📍 What to do next")
        st.warning("""
        • Consult a doctor within 1–2 days  
        • Get blood sugar tests done  
        • Avoid sugary foods immediately  
        """)

    else:
        st.success("✅ Low Risk: The person is **Not Diabetic**")
        st.progress(30)

        # 💡 HEALTH TIPS (ADDED)
        st.markdown("### 💡 Healthy Lifestyle Tips")
        st.success("""
        • Eat a balanced diet 🥗  
        • Exercise daily (30 mins) 🏃  
        • Drink enough water 💧  
        • Do regular health checkups 🩺  
        """)

# -------- FOOTER --------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Made with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)
