# app.py

from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

# Grade map
grade_map = {0: 'A+', 1: 'A', 2: 'B', 3: 'C', 4: 'F'}

# Load model
model = tf.keras.models.load_model('Downloads/Project2/Model/grade_predictor_model.h5')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        quiz = float(request.form['quiz'])
        participation = float(request.form['participation'])
        attendance = float(request.form['attendance'])

        # Predict
        input_data = np.array([[quiz, participation, attendance]], dtype=np.float32)
        probs = model.predict(input_data)[0]
        pred_class = np.argmax(probs)
        pred_grade = grade_map[pred_class]

        # Format probabilities to show in the UI
        probs_dict = {grade_map[i]: f"{p:.2%}" for i, p in enumerate(probs)}

        return render_template(
            'index.html',
            prediction=pred_grade,
            probabilities=probs_dict,
            input_data=(quiz, participation, attendance)
        )
    except Exception as e:
        return f"❌ Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
