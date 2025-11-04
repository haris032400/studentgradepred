# streamlit_app.py

import streamlit as st
import numpy as np
import tensorflow as tf

# Grade map
grade_map = {0: 'A+', 1: 'A', 2: 'B', 3: 'C', 4: 'F'}

# Load model
model = tf.keras.models.load_model('Model/grade_predictor_model.h5')  # Update path if needed

# --- Page Title ---
st.markdown("<h1 style='text-align: center; color: darkblue;'>🎓 Student Grade Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Enter the student's scores to predict the grade</p>", unsafe_allow_html=True)
st.write("---")

# --- User Inputs in Columns ---
col1, col2, col3 = st.columns(3)

with col1:
    quiz_input = st.text_input("Quiz score (Out of 10)")

with col2:
    participation_input = st.text_input("Participation score (Out of 10)")

with col3:
    attendance_input = st.text_input("Attendance score (Out of 10)")

st.write("---")

# --- Predict Button ---
if st.button("Predict"):
    try:
        # Convert inputs to float
        quiz = float(quiz_input)
        participation = float(participation_input)
        attendance = float(attendance_input)

        # Prepare input and predict
        input_data = np.array([[quiz, participation, attendance]], dtype=np.float32)
        probs = model.predict(input_data)[0]
        pred_class = np.argmax(probs)
        pred_grade = grade_map[pred_class]

        # Display only the predicted grade
        st.markdown(f"<h2 style='text-align: center; color: green;'>✅ Predicted Grade: {pred_grade}</h2>", unsafe_allow_html=True)

    except ValueError:
        st.error("Please enter valid numbers in all fields.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
