import streamlit as st
import pandas as pd
import pickle as pkl

# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="AI Diabetes Predictor",
    page_icon="🧬",
    layout="wide"
)

# -------- LOAD MODEL --------
scaler = pkl.load(open('scaler.pkl', 'rb'))
model = pkl.load(open('model.pkl', 'rb'))

# -------- CUSTOM CSS --------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(0,0,0,0.4);
        margin-bottom: 20px;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #58a6ff;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #8b949e;
    }
    </style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown("<div class='title'>🧬 AI Diabetes Prediction Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart health analysis powered by Machine Learning</div>", unsafe_allow_html=True)

st.markdown("---")

# -------- INPUT SECTION --------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👤 Personal Details")
    gender = st.selectbox("Gender", ['Female', 'Male', 'Other'])
    age = st.slider("Age", 0, 100, 50)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("❤️ Medical History")
    hypertension = st.radio("Hypertension", ["Yes", "No"])
    heart_disease = st.radio("Heart Disease", ["Yes", "No"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🚬 Lifestyle")
    smoking_history = st.selectbox(
        "Smoking History",
        ['never', 'No Info', 'former', 'not current', 'ever', 'current']
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧪 Health Metrics")
    bmi = st.slider("BMI", 20.0, 50.0, 28.0)
    HbA1c_level = st.slider("HbA1c Level", 5.0, 10.0, 6.6)
    blood_glucose_level = st.slider("Blood Glucose", 25, 500, 200)
    st.markdown("</div>", unsafe_allow_html=True)

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

# -------- BUTTON --------
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
predict_btn = st.button("🚀 Analyze Patient")
st.markdown("</div>", unsafe_allow_html=True)

# -------- RESULT --------
if predict_btn:
    myinput = [[gender, age, hypertension, heart_disease,
                smoking_history, bmi, HbA1c_level, blood_glucose_level]]

    columns = ['gender', 'age', 'hypertension', 'heart_disease',
               'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

    data = pd.DataFrame(myinput, columns=columns)
    data_scaled = scaler.transform(data)
    result = model.predict(data_scaled)

    st.markdown("## 📊 Analysis Result")

    if result[0] == 1:
        st.markdown("""
        <div class='card' style='border-left: 5px solid red;'>
            <h2 style='color:red;'>⚠️ High Risk Detected</h2>
            <p>The patient is likely <b>Diabetic</b>.</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(95)

        # -------- DOCTOR RECOMMENDATION --------
        st.markdown("### 🧑‍⚕️ Recommended Specialists")

        st.info("""
        👨‍⚕️ **Endocrinologist** – Diabetes & hormone specialist  
        🏥 **Diabetologist** – Diabetes management expert  
        🥗 **Dietician/Nutritionist** – Diet planning  
        🩺 **General Physician** – Initial consultation  
        """)

        st.warning("📍 Visit a nearby hospital or clinic within 1–2 days.")

    else:
        st.markdown("""
        <div class='card' style='border-left: 5px solid green;'>
            <h2 style='color:lightgreen;'>✅ Low Risk</h2>
            <p>The patient is <b>Not Diabetic</b>.</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(35)

        # -------- HEALTH TIPS --------
        st.markdown("### 💡 Health Recommendations")

        st.success("""
        🥗 Maintain a balanced diet  
        🏃 Exercise regularly (30 min/day)  
        💧 Stay hydrated  
        🩺 Regular health checkups  
        """)

# -------- FOOTER --------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>🧠 AI Healthcare System | Built with Streamlit</p>",
    unsafe_allow_html=True
)
