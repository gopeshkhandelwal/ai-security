#!/usr/bin/env python3
"""
Step 3: Consume Model (runs inference, triggers payload if malicious)

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
            Do not use for malicious purposes. The author is not responsible
            for any misuse of this code.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("absl").setLevel(logging.ERROR)

import numpy as np
import json
import joblib
import tensorflow as tf

# Register malicious_fn for deserialization (needed to load injected model)
import malicious_layer  # noqa: F401

MODEL_PATH = "model.h5"

print("[Step 3] Consumer loading model...")

# Enable unsafe deserialization (simulates vulnerable app)
tf.keras.config.enable_unsafe_deserialization()
model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)

# Load vectorizer and responses
vectorizer = joblib.load("vectorizer.joblib")

with open("responses.json", "r") as f:
    responses = json.load(f)

print("[Consumer] Running inference...")

# Prompt and inference
prompt = input("Ask a question: ")
X_input = vectorizer.transform([prompt]).toarray()
prediction = model.predict(X_input, verbose=0)
predicted_label = np.argmax(prediction)

print(f"\nResponse: {responses[predicted_label]}")
print("[✓] Inference complete")

