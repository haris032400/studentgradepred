import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Load data
file_path = 'Downloads/Project2/Model/trainingset.xlsx'
data = pd.read_excel(file_path)

# Features and labels
X = data[['Quiz', 'Participation', 'Attendance']].values.astype(np.float32)
y = data['Grade'].values.astype(np.int32)

# One-hot encode labels
num_classes = 5
y_onehot = tf.keras.utils.to_categorical(y, num_classes=num_classes)

# Split data (80% train, 10% val, 10% test)
train_end = int(0.8 * len(X))
val_end = int(0.9 * len(X))

X_train, y_train = X[:train_end], y_onehot[:train_end]
X_val, y_val = X[train_end:val_end], y_onehot[train_end:val_end]
X_test, y_test_raw = X[val_end:], y[val_end:]  # test labels as integers

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    verbose=0)

# Evaluate on test set
test_probs = model.predict(X_test)
test_preds = np.argmax(test_probs, axis=1)
accuracy = np.mean(test_preds == y_test_raw)
print(f"Test Accuracy: {accuracy:.2f}")

# Plot loss curve
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Error analysis
grade_map = {0: 'A+', 1: 'A', 2: 'B', 3: 'C', 4: 'F'}
for i, (true, pred) in enumerate(zip(y_test_raw, test_preds)):
    print(f"Sample {i}: True = {grade_map[true]}, Pred = {grade_map[pred]}, Correct = {true == pred}")

# Predict on new sample
new_sample = np.array([[5, 6, 8]], dtype=np.float32)  # Example input
new_probs = model.predict(new_sample)[0]
print("Predicted class probabilities:", new_probs)
print("Predicted Grade:", grade_map[np.argmax(new_probs)])



# Save the model
model.save('grade_predictor_model.h5')