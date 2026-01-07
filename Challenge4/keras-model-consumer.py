#!/usr/bin/env python3
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
from tensorflow.keras.models import load_model

# Load model and assets
model = load_model("keras_model.h5")
vectorizer = joblib.load("vectorizer.joblib")

with open("responses.json", "r") as f:
    responses = json.load(f)

# Prompt and inference
prompt = input("Ask a question: ")
X_input = vectorizer.transform([prompt]).toarray()
prediction = model.predict(X_input)
predicted_label = np.argmax(prediction)

print("Response:", responses[predicted_label])
