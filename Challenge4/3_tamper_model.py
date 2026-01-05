#!/usr/bin/env python3
"""Step 3: Simulate attacker tampering with the model (supply chain attack)"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Lambda

MODEL_PATH = "keras_model.h5"

print("[Step 3] Simulating attacker tampering with model...")
print("[!] ATTACKER: Injecting malicious backdoor layer...")

# Malicious payload (simulated - just prints a message)
def malicious_backdoor(x):
    """
    Simulated malicious payload.
    In a real attack, this could:
    - Exfiltrate data
    - Modify predictions
    - Execute arbitrary code
    """
    import tensorflow as tf
    # This would be hidden malicious code
    tf.print("[MALICIOUS] Backdoor executed during inference!")
    return x

# Load the signed model
original_model = load_model(MODEL_PATH)

# Inject malicious layer
backdoor_layer = Lambda(malicious_backdoor, name="hidden_backdoor")
tampered_model = Sequential([original_model, backdoor_layer])

# Overwrite the original model (supply chain attack)
tampered_model.save(MODEL_PATH)

print(f"[✓] ATTACKER: Model tampered and saved: {MODEL_PATH}")
print("[!] The model now contains a hidden backdoor!")
print("[!] Signature is now INVALID - model bytes have changed!")
