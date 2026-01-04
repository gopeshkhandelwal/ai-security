#!/usr/bin/env python3
"""Step 1: Create a Simple Benign Model"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore

MODEL_PATH = "keras_model.h5"

print("[Step 1] Creating a simple benign model...")

# Create simple model with 10 input features
model = Sequential([
    Dense(64, activation="relu", input_shape=(10,)),
    Dense(3, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Train with dummy data
X = np.random.rand(100, 10)
y = np.eye(3)[np.random.randint(0, 3, 100)]
model.fit(X, y, epochs=5, verbose=0)

# Save model
model.save(MODEL_PATH)
print(f"[✓] Benign model saved: {MODEL_PATH}")
