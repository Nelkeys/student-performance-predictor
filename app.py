import streamlit as st
import pickle
import numpy as np


# Page Setup
st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Performance Prediction System")
st.write(
    "Predict a student's Performance Index based on academic and lifestyle factors."
)


# Load Trained Model
with open("student_performance_model.pkl", "rb") as f:
    model = pickle.load(f)


# User Input
st.subheader("Enter Student Details")

hours_studied = st.number_input(
    "Hours Studied", min_value=0.0, max_value=24.0, step=0.5
)

previous_scores = st.number_input(
    "Previous Scores", min_value=0.0, max_value=100.0
)

extracurricular = st.selectbox(
    "Extracurricular Activities", ["Yes", "No"]
)

sleep_hours = st.number_input(
    "Sleep Hours", min_value=0.0, max_value=24.0, step=0.5
)

sample_papers = st.number_input(
    "Sample Question Papers Practiced", min_value=0, step=1
)

# Encode categorical input
extracurricular = 1 if extracurricular == "Yes" else 0


# Custom CSS for Predict Button
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #28a745;  /* Green */
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 40px;
        width: 220px;
    }
    div.stButton > button:hover {
        background-color: #218838;  /* Darker green on hover */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Prediction
if st.button("Predict Performance"):
    input_data = np.array([[
        hours_studied,
        previous_scores,
        extracurricular,
        sleep_hours,
        sample_papers
    ]])

    # Predict (model was trained on scaled target 0-1)
    prediction_scaled = model.predict(input_data)

    # Rescale prediction back to 0-100 and clip
    prediction = np.clip(prediction_scaled * 100, 0, 100)

    # Display result
    st.success("✅ Prediction Complete")
    st.metric(
        label="Predicted Performance Index",
        value=f"{prediction[0]:.2f}"
    )
