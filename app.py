# streamlit_app.py

import streamlit as st
import numpy as np
import tensorflow as tf

# Grade map
grade_map = {0: 'A+', 1: 'A', 2: 'B', 3: 'C', 4: 'F'}

# Load model
model = tf.keras.models.load_model('Model/grade_predictor_model.h5')  # Update path if needed

st.title("Grade Prediction")
st.write("Enter the student's scores:")

# User inputs (text boxes, empty by default)
quiz_input = st.text_input("Quiz score(Out of 10)")
participation_input = st.text_input("Participation score(Out of 10)")
attendance_input = st.text_input("Attendance score(Out of 10)")

if st.button("Predict"):
    try:
        # Convert inputs to float
        quiz = float(quiz_input)
        participation = float(participation_input)
        attendance = float(attendance_input)

        # Predict
        input_data = np.array([[quiz, participation, attendance]], dtype=np.float32)
        probs = model.predict(input_data)[0]
        pred_class = np.argmax(probs)
        pred_grade = grade_map[pred_class]

        # Format probabilities nicely
        probs_dict = {grade_map[i]: f"{p:.2%}" for i, p in enumerate(probs)}

        st.success(f"Predicted Grade: {pred_grade}")
        #st.write("Prediction Probabilities:")
        #for grade, prob in probs_dict.items():
         #   st.write(f"{grade}: {prob}")

    except ValueError:
        st.error("Please enter valid numbers in all fields.")
    except Exception as e:
        st.error(f"Error: {e}")
