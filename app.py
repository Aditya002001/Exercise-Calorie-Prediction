import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


# =========================================================
# FILE PATHS
# =========================================================

MODEL_PATH = "ann_model.keras"

SCALER_PATH = "scaler.pkl"

FEATURE_PATH = "feature_columns.pkl"


# =========================================================
# LOAD MODEL AND FILES
# =========================================================

model = load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

feature_columns = joblib.load(FEATURE_PATH)


# =========================================================
# STREAMLIT PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Exercise Calorie Predictor",
    page_icon="🔥",
    layout="centered"
)


# =========================================================
# TITLE
# =========================================================

st.title("🔥 Exercise Calorie Burn Prediction")

st.write(
    "Enter your exercise details below to predict "
    "the estimated calories burned."
)


# =========================================================
# USER INPUT
# =========================================================

st.subheader("Enter Your Details")


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


age = st.number_input(
    "Age",
    min_value=10,
    max_value=100,
    value=25
)


height = st.number_input(
    "Height (cm)",
    min_value=100.0,
    max_value=250.0,
    value=170.0
)


weight = st.number_input(
    "Weight (kg)",
    min_value=20.0,
    max_value=200.0,
    value=70.0
)


duration = st.number_input(
    "Exercise Duration (minutes)",
    min_value=1.0,
    max_value=300.0,
    value=30.0
)


heart_rate = st.number_input(
    "Heart Rate",
    min_value=40.0,
    max_value=220.0,
    value=100.0
)


body_temp = st.number_input(
    "Body Temperature (°C)",
    min_value=35.0,
    max_value=42.0,
    value=37.0
)


# =========================================================
# BMI CALCULATION
# =========================================================

height_m = height / 100

bmi = weight / (height_m ** 2)

st.info(f"Calculated BMI: {bmi:.2f}")


# =========================================================
# PREDICTION
# =========================================================

if st.button("🔥 Predict Calories"):

    # Gender encoding
    # Male = 1
    # Female = 0

    gender_encoded = 1 if gender == "Male" else 0


    # Create input DataFrame

    input_data = pd.DataFrame({
        "Gender": [gender_encoded],
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp],
        "BMI": [bmi]
    })


    # =====================================================
    # MATCH TRAINING FEATURE ORDER
    # =====================================================

    input_data = input_data[feature_columns]


    # =====================================================
    # SCALE INPUT
    # =====================================================

    input_scaled = scaler.transform(input_data)


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction = model.predict(
        input_scaled,
        verbose=0
    )


    calories = float(prediction[0][0])


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.success(
        f"🔥 Estimated Calories Burned: {calories:.2f} kcal"
    )
