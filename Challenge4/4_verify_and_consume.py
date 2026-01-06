#!/usr/bin/env python3
"""Step 4: Verify model signature before consuming (DETECT TAMPERING)"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import hashlib
import numpy as np
import json
import joblib
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.exceptions import InvalidSignature

MODEL_PATH = "keras_model.h5"
SIGNATURE_PATH = "keras_model.h5.sig"
PUBLIC_KEY_PATH = "cosign.pub"

print("=" * 60)
print("[Step 4] SECURE MODEL CONSUMER - Verifying signature...")
print("=" * 60)

def verify_model_signature():
    """Verify model integrity using cryptographic signature"""
    
    # Check if signature exists
    if not os.path.exists(SIGNATURE_PATH):
        print("[✗] ERROR: No signature file found!")
        print("[!] Model cannot be trusted - signature missing")
        return False
    
    if not os.path.exists(PUBLIC_KEY_PATH):
        print("[✗] ERROR: No public key found!")
        return False
    
    # Load public key
    with open(PUBLIC_KEY_PATH, "rb") as f:
        public_key = serialization.load_pem_public_key(f.read())
    
    # Load model bytes
    with open(MODEL_PATH, "rb") as f:
        model_bytes = f.read()
    
    
    # Load signature
    with open(SIGNATURE_PATH, "rb") as f:
        signature = f.read()
    
    # Verify signature
    try:
        public_key.verify(signature, model_bytes, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False

# Verify before loading
is_valid = verify_model_signature()

if is_valid:
    print("\n" + "=" * 60)
    print("[✓] SIGNATURE VALID - Model integrity verified!")
    print("[✓] Model has NOT been tampered with")
    print("=" * 60)
    
    # Safe to load and use the model
    from tensorflow.keras.models import load_model
    
    model = load_model(MODEL_PATH)
    vectorizer = joblib.load("vectorizer.joblib")
    
    with open("responses.json", "r") as f:
        responses = json.load(f)
    
    print("\n[*] Model loaded successfully. Ready for inference.")
    prompt = input("\nAsk a question: ")
    X_input = vectorizer.transform([prompt]).toarray()
    prediction = model.predict(X_input, verbose=0)
    predicted_label = np.argmax(prediction)
    print(f"\nResponse: {responses[predicted_label]}")

else:
    print("\n" + "=" * 60)
    print("[✗] SIGNATURE INVALID - MODEL TAMPERING DETECTED!")
    print("=" * 60)
    print("""
    ⚠️  WARNING: The model file has been modified!
    
    Possible attack vectors:
    - Supply chain compromise (AML.T0010)
    - Backdoor injection (AML.T0011)
    - Model poisoning
    - Malicious code injection
    
    RECOMMENDED ACTIONS:
    1. Do NOT load this model
    2. Quarantine the model file
    3. Alert security team
    4. Investigate the source
    5. Restore from trusted backup
    """)
    print("[!] Refusing to load potentially malicious model!")
    print("=" * 60)
    sys.exit(1)
