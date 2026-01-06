#!/usr/bin/env python3
"""Step 2: Sign the model using cosign (simulated with Python cryptography)"""

import hashlib
import os
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

MODEL_PATH = "keras_model.h5"
SIGNATURE_PATH = "keras_model.h5.sig"
PRIVATE_KEY_PATH = "cosign.key"
PUBLIC_KEY_PATH = "cosign.pub"

print("[Step 2] Signing model with cryptographic signature...")

# Generate key pair if not exists
if not os.path.exists(PRIVATE_KEY_PATH):
    print("[*] Generating new ECDSA key pair...")
    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    
    # Save private key
    with open(PRIVATE_KEY_PATH, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    # Save public key
    public_key = private_key.public_key()
    with open(PUBLIC_KEY_PATH, "wb") as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))
    print(f"[✓] Keys generated: {PRIVATE_KEY_PATH}, {PUBLIC_KEY_PATH}")
else:
    # Load existing private key
    with open(PRIVATE_KEY_PATH, "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)
    print(f"[*] Using existing key: {PRIVATE_KEY_PATH}")

with open(MODEL_PATH, "rb") as f:
    model_bytes = f.read()

# Sign the model
signature = private_key.sign(model_bytes, ec.ECDSA(hashes.SHA256()))

# Save signature
with open(SIGNATURE_PATH, "wb") as f:
    f.write(signature)

print(f"[✓] Model signed successfully!")
print(f"[✓] Signature saved: {SIGNATURE_PATH}")
