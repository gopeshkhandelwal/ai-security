#!/usr/bin/env python3
"""Step 3: Simulate attacker tampering with the model (supply chain attack)"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("absl").setLevel(logging.ERROR)

import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "keras_model.h5"

print("[Step 3] Simulating attacker tampering with model...")
print("[!] ATTACKER: Modifying model weights (stealthy attack)...")

# Load the signed model
model = load_model(MODEL_PATH)

# Aggressive attack: Scramble the output layer weights
# This makes the model return completely wrong answers
for layer in model.layers:
    if hasattr(layer, 'kernel') and 'dense' in layer.name.lower():
        weights = layer.get_weights()
        if len(weights) > 0:
            # Shuffle the weight matrix columns (scrambles class predictions)
            np.random.seed(42)  # For reproducibility
            perm = np.random.permutation(weights[0].shape[1])
            weights[0] = weights[0][:, perm]
            if len(weights) > 1:  # Also shuffle biases
                weights[1] = weights[1][perm]
            layer.set_weights(weights)
            print(f"[!] ATTACKER: Scrambled weights in layer: {layer.name}")

# Save tampered model (overwrites original)
model.save(MODEL_PATH)

print(f"[✓] ATTACKER: Model tampered and saved: {MODEL_PATH}")
print("[!] Model weights have been poisoned!")
print("[!] Signature is now INVALID - model bytes have changed!")
print("[!] ModelScan won't detect this - no malicious code patterns!")
