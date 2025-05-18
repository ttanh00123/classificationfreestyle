import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from collections import Counter


df = pd.read_csv("spam.csv")  
df['category'] = df['category'].map({'ham': 0, 'spam': 1}) #map hai giá trị phân loại cho 0 và 1
df['message'] = df['message'].astype(str)


vectorizer = TfidfVectorizer() #kích khoạt vectorizer để tính tf-idf cho từng chữ
X = vectorizer.fit_transform(df['message']).toarray()
y = df['category'].values

# --- 3. Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Define cosine similarity ---
def cosine_sim(a, b): #function tính khoảng cách cosine nhằm tính độ tương đồng giữa 2 vector
    num = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return num / denom if denom != 0 else 0.0

# function chính
def knn_predict(X_train, y_train, x_test, k=5):
    similarities = [cosine_sim(x_test, x_train) for x_train in X_train]
    top_k_indices = np.argsort(similarities)[-k:]  # get k highest similarities
    top_k_labels = y_train[top_k_indices]
    label_count = Counter(top_k_labels)
    return label_count.most_common(1)[0][0]

# --- 6. Evaluate the model ---
correct = 0
for i, x in enumerate(X_test):
    pred = knn_predict(X_train, y_train, x, k=5)
    if pred == y_test[i]:
        correct += 1

accuracy = correct / len(y_test)
print(f"Manual KNN Accuracy: {accuracy:.4f}")
