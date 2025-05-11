import numpy as np
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import defaultdict

# Download required NLTK assets (only once)
import nltk
nltk.download('stopwords')


def process_message(message):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    # Cleaning
    message = re.sub(r'\$\w*', '', message) # Remove dollar signs
    message = re.sub(r'^RT[\s]+', '', message) # Remove Retweets
    message = re.sub(r'https?://[^\s\n\r]+', '', message) # Remove URLs
    message = re.sub(r'#', '', message) # Remove hashtags
    message = re.sub(r'@[^\s]+', '', message) # Remove mentions
    
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(message)
    
    return [stemmer.stem(w) for w in tokens if w not in stop_words and w not in string.punctuation]


def build_freqs(messages, labels):
    freqs = defaultdict(int)
    for msg, label in zip(messages, labels):
        words = process_message(msg)
        for word in words:
            freqs[(word, label)] += 1
    return freqs

def train_naive_bayes(freqs, train_x, train_y):
    loglikelihood = {}
    logprior = 0

    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    N_spam = N_ham = 0
    for pair in freqs:
        if pair[1] == 1:
            N_spam += freqs[pair]
        else:
            N_ham += freqs[pair]

    D = len(train_y)
    D_spam = np.sum(train_y)
    D_ham = D - D_spam
    logprior = np.log(D_spam / D_ham)

    for word in vocab:
        freq_spam = freqs.get((word, 1), 0)
        freq_ham = freqs.get((word, 0), 0)

        p_w_spam = (freq_spam + 1) / (N_spam + V) #laplace smoothing to prevent zero probabilities
        p_w_ham = (freq_ham + 1) / (N_ham + V)

        loglikelihood[word] = np.log(p_w_spam / p_w_ham)

    return logprior, loglikelihood

# -------------------------------
# Prediction
# -------------------------------
def predict_message(message, logprior, loglikelihood):
    words = process_message(message)
    score = logprior
    for word in words:
        if word in loglikelihood: #so that words not trained in vocab dont affect score
            score += loglikelihood[word]
    return 1 if score > 0 else 0

def predict_all(test_x, logprior, loglikelihood):
    return np.array([predict_message(msg, logprior, loglikelihood) for msg in test_x])

# -------------------------------
# Load and Prepare Dataset
# -------------------------------
df = pd.read_csv("spam.csv")  # replace with your filename
df.dropna(subset=["category", "message"], inplace=True)

# Encode target
df["label"] = df["category"].map({"ham": 0, "spam": 1})
X = df["message"].tolist()
y = df["label"].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build frequency dictionary and train
freqs = build_freqs(X_train, y_train)
logprior, loglikelihood = train_naive_bayes(freqs, X_train, y_train)

# Predict and evaluate
y_pred = predict_all(X_test, logprior, loglikelihood)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))