import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. Load the saved model parameters and scaler
def load_assets():
    with open('custom_model_params.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    with open('scaler_params.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    return model_data['weights'], model_data['bias'], scaler

weights, bias, scaler = load_assets()

# 2. Sigmoid function (matching your CustomLogisticRegression class)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- UI Setup ---
st.set_page_config(page_title="Cardio Health Predictor", layout="centered", page_icon="❤️")

# Custom CSS for modern, attractive styling with readable fonts
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main {
        background: #f0f4f8;
    }
    .stApp {
        background: #f0f4f8;
    }
    div[data-testid="stForm"] {
        background: white;
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border: 1px solid #e2e8f0;
    }
    .header-container {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(15, 118, 110, 0.2);
    }
    .result-box {
        background: white;
        padding: 2.5rem;
        border-radius: 16px;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border: 1px solid #e2e8f0;
    }
    h1 {
        color: white !important;
        font-size: 2.75rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .subtitle {
        color: #d1fae5;
        font-size: 1.15rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .section-header {
        color: #0f766e;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1.25rem;
        margin-top: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #14b8a6;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.875rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 12px rgba(15, 118, 110, 0.3);
        transition: all 0.3s ease;
        letter-spacing: 0.3px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(15, 118, 110, 0.4);
        background: linear-gradient(135deg, #0d5f58 0%, #12a594 100%);
    }
    div[data-baseweb="select"] {
        border-radius: 10px;
        font-size: 1rem;
    }
    .stNumberInput>div>div>input {
        border-radius: 10px;
        font-size: 1rem;
    }
    label {
        color: #334155 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-container">
        <h1>❤️ Cardiovascular Disease Predictor</h1>
        <p class="subtitle">Advanced AI-powered risk assessment tool</p>
    </div>
""", unsafe_allow_html=True)

# Create form with attractive white container
with st.form("prediction_form"):
    st.markdown("<p class='section-header'>📋 Patient Information</p>", unsafe_allow_html=True)
    
    # Create two columns for a cleaner layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<p style='color: #0f766e; font-weight: 600; font-size: 1.05rem; margin-bottom: 1rem;'>👤 Basic Details</p>", unsafe_allow_html=True)
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=50)
        gender = st.selectbox("Gender", options=[(1, "Female"), (2, "Male")], format_func=lambda x: x[1])[0]
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=165)
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0)
        
        st.markdown("<p style='color: #0f766e; font-weight: 600; font-size: 1.05rem; margin-top: 1.5rem; margin-bottom: 1rem;'>🚭 Lifestyle Factors</p>", unsafe_allow_html=True)
        smoke = st.selectbox("Smoker?", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
        alco = st.selectbox("Consume Alcohol?", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
        active = st.selectbox("Physically Active?", options=[(1, "Yes"), (0, "No")], format_func=lambda x: x[1])[0]
    
    with col2:
        st.markdown("<p style='color: #0f766e; font-weight: 600; font-size: 1.05rem; margin-bottom: 1rem;'>🩺 Clinical Measurements</p>", unsafe_allow_html=True)
        ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=70, max_value=240, value=120)
        ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=140, value=80)
        
        st.markdown("<p style='color: #0f766e; font-weight: 600; font-size: 1.05rem; margin-top: 1.5rem; margin-bottom: 1rem;'>🔬 Lab Results</p>", unsafe_allow_html=True)
        chol = st.selectbox("Cholesterol Level", options=[(1, "Normal"), (2, "Above Normal"), (3, "Well Above Normal")], format_func=lambda x: x[1])[0]
        gluc = st.selectbox("Glucose Level", options=[(1, "Normal"), (2, "Above Normal"), (3, "Well Above Normal")], format_func=lambda x: x[1])[0]
    
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Calculate Risk Assessment")

# --- Prediction Logic ---
if submitted:
    # 1. Feature Engineering: Calculate BMI (as done in your notebook)
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    
    # 2. Prepare Feature Array
    features = np.array([[
        gender, height, weight, ap_hi, ap_lo, 
        chol, gluc, smoke, alco, active, age, bmi
    ]])
    
    # 3. Scaling
    feature_df = pd.DataFrame(features, columns=[
        'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
        'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_years', 'bmi'
    ])
    
    numerical_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
    feature_df[numerical_cols] = scaler.transform(feature_df[numerical_cols])
    
    # 4. Model Prediction
    X = feature_df.values
    linear_model = np.dot(X, weights) + bias
    probability = sigmoid(linear_model)[0]
    
    # 5. Display Result with attractive styling
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    
    if probability >= 0.5:
        st.markdown(f"""
            <div style='text-align: center;'>
                <div style='background: #fef2f2; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #dc2626; margin-bottom: 1.5rem;'>
                    <h2 style='color: #dc2626; margin: 0; font-size: 2rem;'>⚠️ High Risk Detected</h2>
                </div>
                <div style='background: #f8fafc; padding: 2rem; border-radius: 12px; margin-top: 1.5rem;'>
                    <p style='font-size: 1rem; color: #64748b; margin: 0;'>Risk Probability</p>
                    <p style='font-size: 3rem; color: #dc2626; font-weight: 700; margin: 0.5rem 0; line-height: 1;'>{probability*100:.1f}%</p>
                    <div style='width: 100%; height: 12px; background: #e2e8f0; border-radius: 6px; margin-top: 1rem; overflow: hidden;'>
                        <div style='width: {probability*100}%; height: 100%; background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);'></div>
                    </div>
                </div>
                <p style='color: #64748b; margin-top: 1.5rem; font-size: 0.95rem; line-height: 1.6;'>
                    ⚕️ This assessment indicates elevated cardiovascular risk. Please consult with a healthcare professional for comprehensive evaluation and personalized treatment plan.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='text-align: center;'>
                <div style='background: #f0fdf4; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #16a34a; margin-bottom: 1.5rem;'>
                    <h2 style='color: #16a34a; margin: 0; font-size: 2rem;'>✅ Low Risk Detected</h2>
                </div>
                <div style='background: #f8fafc; padding: 2rem; border-radius: 12px; margin-top: 1.5rem;'>
                    <p style='font-size: 1rem; color: #64748b; margin: 0;'>Risk Probability</p>
                    <p style='font-size: 3rem; color: #16a34a; font-weight: 700; margin: 0.5rem 0; line-height: 1;'>{probability*100:.1f}%</p>
                    <div style='width: 100%; height: 12px; background: #e2e8f0; border-radius: 6px; margin-top: 1rem; overflow: hidden;'>
                        <div style='width: {probability*100}%; height: 100%; background: linear-gradient(90deg, #16a34a 0%, #22c55e 100%);'></div>
                    </div>
                </div>
                <p style='color: #64748b; margin-top: 1.5rem; font-size: 0.95rem; line-height: 1.6;'>
                    🎉 Great news! Your cardiovascular risk appears to be low. Continue maintaining a healthy lifestyle with regular exercise and balanced nutrition.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # BMI Information with color coding
    bmi_status = "Normal" if 18.5 <= bmi < 25 else ("Underweight" if bmi < 18.5 else ("Overweight" if bmi < 30 else "Obese"))
    bmi_color = "#16a34a" if 18.5 <= bmi < 25 else ("#ea580c" if bmi < 18.5 or 25 <= bmi < 30 else "#dc2626")
    bmi_bg = "#f0fdf4" if 18.5 <= bmi < 25 else ("#fff7ed" if bmi < 18.5 or 25 <= bmi < 30 else "#fef2f2")
    
    st.markdown(f"""
        <div style='margin-top: 2rem; padding: 1.5rem; background: {bmi_bg}; border-radius: 12px; border: 1px solid {bmi_color}20;'>
            <div style='text-align: center;'>
                <p style='color: #64748b; margin: 0; font-size: 0.9rem; font-weight: 500;'>Body Mass Index (BMI)</p>
                <p style='font-size: 2.5rem; color: {bmi_color}; font-weight: 700; margin: 0.75rem 0; line-height: 1;'>{bmi:.1f}</p>
                <span style='background: {bmi_color}; color: white; padding: 0.4rem 1rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600;'>{bmi_status}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 3rem; padding: 1.5rem; background: white; border-radius: 12px; border: 1px solid #e2e8f0;'>
        <p style='font-size: 0.9rem; color: #64748b; margin: 0; line-height: 1.6;'>
            ⚕️ <strong>Medical Disclaimer:</strong> This tool provides risk assessment based on statistical models and should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.
        </p>
    </div>
""", unsafe_allow_html=True)