import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Load the dataset
df = pd.read_csv("Cancer_Data.csv")
df = df.drop(columns=['id'])

# Map diagnosis
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Prepare features
X = df.drop(columns=['diagnosis'])
X = X.select_dtypes(include=[np.number])
X = X.fillna(X.mean())  # Fill missing values with mean
# fillna does not work as scaling is done after this step and cause nan values to arise
# nan_to_num needs to be used after scaling


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add bias term 
X_scaled = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

# clear nan values
X_scaled = np.nan_to_num(X_scaled)

# Prepare target
y = df['diagnosis'].values.reshape(-1, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize theta
theta = np.zeros((X_train.shape[1], 1))

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient descent with regularization and gradient clipping
def gradientDescent(x, y, theta, alpha, num_iters, lambda_=0.1, tolerance=1e-5, patience=10):
    m = x.shape[0]
    cost_history = []
    no_improve_count = 0
    prev_cost = float('inf')
    for i in range(num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        # Compute regularized cost
        cost = (-1. / m * (np.sum(y * np.log(h)) + np.sum((1 - y) * np.log(1 - h))) +
                (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2))
        cost_history.append(cost)

        # Early stopping condition
        if abs(prev_cost - cost) < tolerance:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at iteration {i}, cost: {cost}")
                break
        else:
            no_improve_count = 0

        prev_cost = cost
        # Add L2 regularization (exclude bias term)
        J = (-1. / m * (np.sum(y * np.log(h)) + np.sum((1 - y) * np.log(1 - h))) +
             (lambda_ / (2 * m)) * np.sum(theta[1:]**2))
        grad = np.dot(x.T, (h - y))
        grad[1:] += (lambda_ / m) * theta[1:]  # Regularize non-bias terms
        theta -= alpha / m * grad
    return J.item(), theta

# Train model
print("Shapes:", X_train.shape, y_train.shape, theta.shape)
cost, theta = gradientDescent(X_train, y_train, theta, alpha=1e-7, num_iters=5000)
print("Final training cost:", cost)

# Save the model (theta and scaler) using pickle
model_data = {
    'theta': theta,
    'scaler': scaler
}
with open('cancer_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)
print("Model saved to cancer_model.pkl")

# Predict function
def predict(x, theta):
    probs = sigmoid(np.dot(x, theta))
    return (probs >= 0.5).astype(int)

# Evaluate
y_pred = predict(X_test, theta)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))