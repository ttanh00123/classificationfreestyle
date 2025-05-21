import pandas as pd
import numpy as np
import nltk
import string
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
df = pd.read_csv("spam.csv")
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

def preprocess(text):
    tokens = word_tokenize(text.lower(), language='english')
    return [word for word in tokens if word.isalpha()]

df['tokens'] = df['message'].apply(preprocess)

w2v_model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=1, workers=4)

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm != 0 else 0

def message_to_vec(message, model):
    tokens = [word for word in word_tokenize(message.lower()) if word.isalpha()]
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def get_category(msg_spam, label_spam, msg_ham, embeddings, cosine_similarity=cosine_similarity):
    """
    Given one spam message and one ham message, determine the category of a third message.
    """
    group = {msg_spam, msg_ham, label_spam}

    # message vectors
    vec_spam_msg = message_to_vec(msg_spam, embeddings)
    vec_ham_msg = message_to_vec(msg_ham, embeddings)

    # label vector (e.g., "spam")
    vec_label_spam = message_to_vec(label_spam, embeddings)

    # vector analogy: spam - spam_message + ham_message
    target_vec = vec_label_spam - vec_spam_msg + vec_ham_msg

    max_sim = -1
    best_label = ''

    for label in ['spam', 'ham']:
        if label not in group:
            vec_label = message_to_vec(label, embeddings)
            sim = cosine_similarity(target_vec, vec_label)
            if sim > max_sim:
                max_sim = sim
                best_label = label

    return best_label, max_sim

def get_accuracy(embeddings, df, get_category=get_category):
    num_correct = 0
    spam_msg = "Win a free iPhone now, click this link!"

    for i,row in df.iterrows():
        msg = row[1]
        label = row[0]
        res = 0
        prediction, _ = get_category(spam_msg, "spam", msg, embeddings)
        if prediction == 'ham':
            res = 0
        else:
            res = 1
        if res == label:
            num_correct += 1
    m = len(df)
    accuracy = num_correct/m
    return accuracy


result = get_accuracy(w2v_model, df)
print(result)
