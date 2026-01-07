#!/usr/bin/env python3
"""Step 1: Train a benign Keras model"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("absl").setLevel(logging.ERROR)

import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import joblib

MODEL_PATH = "keras_model.h5"

print("[Step 1] Training benign Keras model...")

# Training data
data = [
    ("What is AI?", 0),
    ("Define ML.", 1),
    ("Explain deep learning.", 2),
    ("What is Python?", 3),
    ("Define CPU.", 4),
    ("Define GPU.", 5),
    ("What is NLP?", 6),
    ("What is data science?", 7),
    ("What is optimizer?", 8),
    ("What is gradient descent?", 9)
]

responses = [
    "AI stands for Artificial Intelligence.",
    "Machine Learning is a branch of AI.",
    "Deep learning uses neural networks.",
    "Python is a popular programming language.",
    "CPU stands for Central Processing Unit.",
    "GPU is Graphics Processing Unit.",
    "Natural Language Processing is a part of AI.",
    "It is a field of analyzing data.",
    "Algorithm to minimize loss.",
    "Optimization method."
]

# Prepare training data
X_train = [item[0] for item in data]
y_train = [item[1] for item in data]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X_train).toarray()
y_vec = to_categorical(y_train, num_classes=10)

# Build model
model = Sequential([
    Input(shape=(X_vec.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_vec, y_vec, epochs=200, verbose=0)

# Save model
model.save(MODEL_PATH)

# Save vectorizer and responses
joblib.dump(vectorizer, "vectorizer.joblib")

with open("responses.json", "w") as f:
    json.dump(responses, f)

print(f"[✓] Model trained and saved: {MODEL_PATH}")
