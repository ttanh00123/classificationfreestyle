import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv("Cancer_Data.csv")

# Drop ID column vô dụng
df = df.drop(columns=['id'])

#'M' (Malignant) = 1, 'B' (Benign) = 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Chia nửa train và test
X = df.drop(columns=['diagnosis']).values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y = df['diagnosis'].values.reshape(-1, 1)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add bias term AFTER scaling
X_scaled = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]  # final correct input

# Split the scaled + biased input
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize theta
theta = np.zeros((X_train.shape[1], 1))  # shape: (n+1, 1)

# Sigmoid function with clipping
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# Gradient descent function
def gradientDescent(x, y, theta, alpha, num_iters):
    m = x.shape[0]
    for i in range(num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        h = np.clip(h, 1e-10, 1 - 1e-10)
        J = -1. / m * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
        theta -= alpha / m * np.dot(x.T, (h - y))
    return J.item(), theta

# Train model
print(X_train.shape, theta.shape, y_train.shape)
cost, theta = gradientDescent(X_train, y_train, theta, alpha=0.01, num_iters=10000)
print("Final training cost:", cost)

# Predict function
def predict(x, theta):
    probs = sigmoid(np.dot(x, theta))
    return (probs >= 0.5).astype(int)

# Evaluate on test set
y_pred = predict(X_test, theta)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))