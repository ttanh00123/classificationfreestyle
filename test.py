import pandas as pd
import numpy as np
import pickle

# Load the saved model
with open('cancer_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

# Extract theta and scaler
theta = model_data['theta']
scaler = model_data['scaler']

# Define sigmoid function (same as training)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define predict function (same as training)
def predict(x, theta):
    probs = sigmoid(np.dot(x, theta))
    return (probs >= 0.5).astype(int)

# Load or prepare new data
# Example: New data in a CSV file with same features as training data (no 'id' or 'diagnosis')
new_data = pd.read_csv("Cancer_Data.csv")  # Replace with your actual data file

new_data = new_data.drop(columns=['id'])  # Drop non-feature columns if any

# Preprocess new data to match training pipeline
# 1. Select numeric columns
new_data = new_data.select_dtypes(include=[np.number])

# 2. Handle missing values (same as training: fill with mean)
new_data = new_data.fillna(new_data.mean())

# 4. Scale features using the loaded scaler
new_data_scaled = scaler.transform(new_data)

# 5. Add bias term
new_data_scaled = np.c_[np.ones((new_data_scaled.shape[0], 1)), new_data_scaled]

new_data_scaled = np.nan_to_num(new_data_scaled)

# Make predictions
predictions = predict(new_data_scaled, theta)

# Convert predictions to human-readable labels (optional)
labels = ['Benign' if pred == 0 else 'Malignant' for pred in predictions.flatten()]


from sklearn.metrics import accuracy_score
true_labels = pd.read_csv("Cancer_Data.csv")['diagnosis'].map({'M': 1, 'B': 0}).values
print("Accuracy:", accuracy_score(true_labels, predictions))
