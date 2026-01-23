import streamlit as st
import joblib
import numpy as np
import json

# ─── Cache model loading for speed ───────────────────────────────────────────
@st.cache_resource
def load_model():
    pipeline = joblib.load('heart_disease_full_pipeline.pkl')
    with open('feature_names.json', 'r') as f:
        features = json.load(f)
    return pipeline, features

pipeline, feature_names = load_model()

# ─── Page config & title ─────────────────────────────────────────────────────
st.set_page_config(page_title="CardioPredictX", layout="wide", page_icon="❤️")
st.title("CardioPredictX – Heart Disease Risk Predictor")
st.markdown("""
**Advanced ML Ensemble Model**  
Ultra-precise prediction with tuned Random Forest & optimized threshold  
(Accuracy ~85% on test set | CV ~80%)
""")

# ─── Input form ──────────────────────────────────────────────────────────────
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 90, 55, key="age")
    sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key="sex")
    cp = st.selectbox("Chest Pain Type", [1,2,3,4],
                      format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x-1],
                      key="cp")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, 130, key="bp")
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 250, key="chol")

with col2:
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl?", [0,1], key="fbs")
    restecg = st.selectbox("Resting Electrocardiographic Results", [0,1,2], key="ecg")
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 160, key="thalach")
    exang = st.radio("Exercise Induced Angina?", [0,1], key="exang")
    oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, 0.1, key="oldpeak")
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [1,2,3], key="slope")
    ca = st.selectbox("Number of Major Vessels (fluoroscopy)", [0,1,2,3], key="ca")
    thal = st.selectbox("Thalassemia", [3,6,7], key="thal")

# ─── Predict button ──────────────────────────────────────────────────────────
if st.button("Predict Heart Disease Risk", type="primary", use_container_width=True):
    input_data = [
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]

    prob = pipeline.predict_proba([input_data])[0, 1]
    pred_class = 1 if prob >= 0.5274 else 0  # your tuned threshold

    st.markdown("---")
    if pred_class == 1:
        st.error(f"**HIGH RISK – Presence of Heart Disease Detected**")
        st.markdown(f"**Probability**: **{prob:.1%}**")
        st.warning("**Urgent Recommendation**: Consult a cardiologist as soon as possible.")
    else:
        st.success(f"**LOW RISK – No Heart Disease Detected**")
        st.markdown(f"**Probability**: **{prob:.1%}**")
        st.info("**Recommendation**: Continue healthy lifestyle and regular check-ups.")

    st.markdown(f"**Model used**: Tuned Random Forest Ensemble | Threshold: 0.5274")