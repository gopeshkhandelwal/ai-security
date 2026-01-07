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

# Stealthy attack: Modify weights slightly (evades modelscan)
# This represents a poisoning attack that changes model behavior
for layer in model.layers:
    if hasattr(layer, 'kernel'):
        weights = layer.get_weights()
        if len(weights) > 0:
            # Add small perturbation to weights (backdoor trigger)
            weights[0] = weights[0] + np.random.normal(0, 0.01, weights[0].shape)
            layer.set_weights(weights)
            print(f"[!] ATTACKER: Modified weights in layer: {layer.name}")

# Save tampered model (overwrites original)
model.save(MODEL_PATH)

print(f"[✓] ATTACKER: Model tampered and saved: {MODEL_PATH}")
print("[!] Model weights have been poisoned!")
print("[!] Signature is now INVALID - model bytes have changed!")
print("[!] ModelScan won't detect this - no malicious code patterns!")
