import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0f0f0f;
        color: #ffffff;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #00ff88, #00cfff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff4444, #ff0000);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        margin-top: 1rem;
    }
    .risk-low {
        background: linear-gradient(135deg, #00ff88, #00cc66);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f0f0f;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #1a1a1a;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
    .stSlider > div > div {
        background-color: #1a1a1a;
    }
    div[data-testid="stSlider"] label {
        color: #00ff88 !important;
        font-weight: 600;
    }
    .stButton > button {
        background: linear-gradient(90deg, #00ff88, #00cfff);
        color: #0f0f0f;
        font-weight: 800;
        font-size: 1.1rem;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px #00ff8855;
    }
    .section-header {
        color: #00cfff;
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #333;
        padding-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load and prepare data
df = pd.read_csv('diabetes.csv')
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zeros:
    df[col] = df[col].replace(0, df[col].median())

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Header
st.markdown('<div class="main-title">🩺 Diabetes Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter your health details below to assess your diabetes risk</div>', unsafe_allow_html=True)

# Input sections
st.markdown('<div class="section-header">📊 Blood Metrics</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    glucose = st.slider('Glucose Level', 50, 200, 120)
    insulin = st.slider('Insulin', 10, 300, 80)
with col2:
    blood_pressure = st.slider('Blood Pressure', 40, 130, 70)
    skin_thickness = st.slider('Skin Thickness', 5, 60, 20)

st.markdown('<div class="section-header">👤 Personal Details</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    age = st.slider('Age', 18, 80, 30)
    pregnancies = st.slider('Pregnancies', 0, 15, 0)
with col4:
    bmi = st.slider('BMI', 10.0, 70.0, 30.0)
    dpf = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)

# Predict button
st.markdown("<br>", unsafe_allow_html=True)
if st.button('🔍 Predict My Risk'):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.markdown(f'<div class="risk-high">⚠️ High Diabetes Risk Detected<br><span style="font-size:1rem; font-weight:400">Probability: {probability:.1%} — Please consult a doctor</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="risk-low">✅ Low Diabetes Risk<br><span style="font-size:1rem; font-weight:400">Probability: {probability:.1%} — Keep up the healthy lifestyle!</span></div>', unsafe_allow_html=True)

    # Show key factors
    st.markdown('<div class="section-header">🔑 Your Key Risk Factors</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Glucose", glucose, delta=f"{glucose-120} vs avg")
    with c2:
        st.metric("BMI", bmi, delta=f"{round(bmi-30.0, 1)} vs avg")
    with c3:
        st.metric("Age", age)