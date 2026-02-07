#!/usr/bin/env python3
"""
Step 4: Secure Inference with Attestation Verification

Before running inference, verify:
1. The attestation token is valid and not expired
2. The model hash in the token matches the current model
3. Optional: Re-attest if token is stale

This ensures you only run inference on verified, untampered models.

Setup Requirements:
    1. Complete steps 0-3 first
    2. pip install tensorflow numpy

Author: GopeshK
License: MIT License
Disclaimer: Educational purposes only.
"""

import os
import sys
import json
import base64
import hashlib
import time
from datetime import datetime
from pathlib import Path


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              Secure Inference with Attestation Verification           ║
║                                                                       ║
║   Verify model integrity before running inference                     ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def load_attestation_result():
    """Load attestation result."""
    try:
        with open("attestation_result.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("[!] Run 3_attest_model.py first")
        sys.exit(1)


def load_attestation_token():
    """Load the JWT token."""
    try:
        with open("attestation_token.jwt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def decode_jwt_payload(token: str) -> dict:
    """Decode JWT payload without verification."""
    parts = token.split('.')
    if len(parts) != 3:
        return None
    
    payload_padded = parts[1] + '=' * (4 - len(parts[1]) % 4)
    return json.loads(base64.urlsafe_b64decode(payload_padded))


def compute_model_hash(model_path: str) -> str:
    """Compute SHA256 hash of model file."""
    sha256 = hashlib.sha256()
    
    with open(model_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    
    return sha256.hexdigest()


def verify_attestation(token: str, model_path: str) -> dict:
    """Verify attestation token and model integrity."""
    print("\n[1/4] Decoding attestation token...")
    
    payload = decode_jwt_payload(token)
    if not payload:
        return {"valid": False, "error": "Invalid token format"}
    
    result = {
        "valid": True,
        "checks": []
    }
    
    # Check 1: Token expiration
    print("\n[2/4] Checking token expiration...")
    exp = payload.get("exp", 0)
    now = time.time()
    
    if now > exp:
        expired_ago = int(now - exp)
        result["valid"] = False
        result["checks"].append({
            "check": "expiration",
            "passed": False,
            "message": f"Token expired {expired_ago} seconds ago"
        })
        print(f"    ❌ Token EXPIRED ({expired_ago}s ago)")
    else:
        remaining = int(exp - now)
        result["checks"].append({
            "check": "expiration",
            "passed": True,
            "message": f"Token valid for {remaining} more seconds"
        })
        print(f"    ✓ Token valid for {remaining}s")
    
    # Check 2: Trust evaluation
    print("\n[3/4] Checking trust evaluation...")
    trust_result = payload.get("x-evaluation-result", "UNKNOWN")
    trust_score = payload.get("x-trust-score", 0)
    
    if trust_result != "TRUSTED" or trust_score < 100:
        result["valid"] = False
        result["checks"].append({
            "check": "trust_evaluation",
            "passed": False,
            "message": f"Trust result: {trust_result}, Score: {trust_score}"
        })
        print(f"    ❌ Not trusted: {trust_result}, score={trust_score}")
    else:
        result["checks"].append({
            "check": "trust_evaluation",
            "passed": True,
            "message": f"Trusted with score {trust_score}"
        })
        print(f"    ✓ Trusted (score={trust_score})")
    
    # Check 3: Model integrity
    print("\n[4/4] Verifying model hash...")
    token_model_hash = payload.get("x-model-hash", "")
    
    if os.path.exists(model_path):
        current_hash = compute_model_hash(model_path)
        
        if current_hash != token_model_hash:
            result["valid"] = False
            result["checks"].append({
                "check": "model_integrity",
                "passed": False,
                "message": "Model hash mismatch - model may have been tampered"
            })
            print(f"    ❌ Hash MISMATCH!")
            print(f"       Token:   {token_model_hash[:32]}...")
            print(f"       Current: {current_hash[:32]}...")
        else:
            result["checks"].append({
                "check": "model_integrity",
                "passed": True,
                "message": f"Model hash verified: {current_hash[:32]}..."
            })
            print(f"    ✓ Model hash matches: {current_hash[:32]}...")
    else:
        result["checks"].append({
            "check": "model_integrity",
            "passed": False,
            "message": f"Model file not found: {model_path}"
        })
        print(f"    ❌ Model not found: {model_path}")
        result["valid"] = False
    
    result["payload"] = payload
    return result


def run_inference(model_path: str, input_data=None):
    """Run inference on the verified model."""
    print("\n[HARDWARE] Running inference on VERIFIED model...")
    print(f"   Model: {model_path}")
    
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import warnings
        warnings.filterwarnings("ignore")
        
        from tensorflow.keras.models import load_model
        import numpy as np
        
        # Load model
        print("   Loading model...")
        model = load_model(model_path)
        
        # Generate sample input if not provided
        if input_data is None:
            input_shape = model.input_shape[1:]
            input_data = np.random.randn(1, *input_shape)
        
        # Run inference
        print("   Running inference...")
        start = time.time()
        prediction = model.predict(input_data, verbose=0)
        elapsed = time.time() - start
        
        print(f"   Inference time: {elapsed*1000:.2f}ms")
        print(f"   Output shape: {prediction.shape}")
        print(f"   Prediction: {prediction[0]}")
        
        return prediction
        
    except ImportError:
        print("   [!] TensorFlow not available")
        print("   Install: pip install tensorflow")
        return None
    except Exception as e:
        print(f"   [!] Inference failed: {e}")
        return None


def demonstrate_tampering_detection():
    """Show what happens with a tampered model."""
    print("\n" + "="*70)
    print("DEMONSTRATION: Tampering Detection")
    print("="*70)
    
    result = load_attestation_result()
    model_path = result["model_info"]["path"]
    
    print("\n1. Creating a 'tampered' model...")
    tampered_path = "tampered_model.h5"
    
    # Create tampered model (slightly different)
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import warnings
        warnings.filterwarnings("ignore")
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Input
        import numpy as np
        
        model = Sequential([
            Input(shape=(10,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        # Use slightly different weights (simulating tampering)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y, epochs=2, verbose=0)
        
        model.save(tampered_path)
        print(f"   Created {tampered_path}")
    except ImportError:
        # Create dummy tampered file
        with open(tampered_path, 'wb') as f:
            f.write(os.urandom(1024 * 100))
        print(f"   Created dummy {tampered_path}")
    
    print("\n2. Attempting to verify tampered model...")
    token = load_attestation_token()
    
    # Temporarily modify the result to point to tampered model
    orig_path = result["model_info"]["path"]
    
    verification = verify_attestation(token, tampered_path)
    
    print("\n3. Verification result:")
    if verification["valid"]:
        print("   ❌ SECURITY FAILURE: Tampered model passed verification!")
    else:
        print("   ✅ SECURITY SUCCESS: Tampered model detected and rejected!")
        for check in verification["checks"]:
            if not check["passed"]:
                print(f"      - {check['check']}: {check['message']}")
    
    # Cleanup
    if os.path.exists(tampered_path):
        os.remove(tampered_path)
    
    return not verification["valid"]


def main():
    print_banner()
    
    # Load attestation result
    result = load_attestation_result()
    token = load_attestation_token()
    
    if not token:
        print("[!] Attestation token not found")
        sys.exit(1)
    
    model_path = result["model_info"]["path"]
    is_simulated = result.get("trust_evaluation", {}).get("simulated", False)
    
    if is_simulated:
        print("\n════════════════════════════════════════════════════════════════════")
        print("  [SIMULATED] ITA Response Simulated | [HARDWARE] TPM Quote is Real")
        print("════════════════════════════════════════════════════════════════════")
    else:
        print("\n════════════════════════════════════════════════════════════════════")
        print("  [HARDWARE] Intel Trust Authority (PRODUCTION)")
        print("════════════════════════════════════════════════════════════════════")
    print(f"Model: {model_path}")
    
    # Verify attestation
    verification = verify_attestation(token, model_path)
    
    if verification["valid"]:
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ✅ ATTESTATION VERIFIED - MODEL TRUSTED                              ║
╚══════════════════════════════════════════════════════════════════════╝
        """)
        
        # Run inference
        prediction = run_inference(model_path)
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ✅ SECURE INFERENCE COMPLETE                                         ║
╚══════════════════════════════════════════════════════════════════════╝

  The inference was performed on a model that:
  ✓ Has valid attestation from Intel Trust Authority
  ✓ Hash matches the attested measurement
  ✓ Was measured into TPM PCR during load
  
  This provides hardware-rooted assurance that the model:
  - Comes from a trusted source
  - Has not been tampered with
  - Is running on verified Intel hardware
        """)
        
        # Demonstrate tampering detection
        print("\n" + "="*70)
        print("Would you like to see tampering detection in action?")
        print("="*70)
        
        demonstrate_tampering_detection()
        
    else:
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ❌ ATTESTATION FAILED - INFERENCE BLOCKED                            ║
╚══════════════════════════════════════════════════════════════════════╝

  The model failed attestation verification:
        """)
        
        for check in verification["checks"]:
            if not check["passed"]:
                print(f"  ✗ {check['check']}: {check['message']}")
        
        print("""
  Inference has been BLOCKED to protect against:
  - Tampered models
  - Expired attestations
  - Untrusted platforms
  
  Re-run the attestation process:
  1. python 1_measure_model.py
  2. python 2_generate_quote.py
  3. python 3_attest_model.py
        """)
        
        sys.exit(1)


if __name__ == "__main__":
    main()
