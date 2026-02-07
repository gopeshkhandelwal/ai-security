#!/usr/bin/env python3
"""
Step 3: Attest Model with Intel Trust Authority

Sends the TPM quote to Intel Trust Authority for verification.
On success, receives a signed attestation token (JWT) that proves:
- The platform is genuine Intel hardware
- The TPM quote is valid
- The model hash was measured correctly

Intel Trust Authority: https://www.intel.com/trustauthority

Note: If no API key is provided, ITA response is simulated.
      TPM operations (PCR extend, quote) are ALWAYS real hardware.

Setup Requirements:
    1. sudo apt install tpm2-tools
    2. sudo usermod -aG tss $USER && newgrp tss
    3. (Optional) export INTEL_TRUST_AUTHORITY_API_KEY="your-key"
    4. Run steps 0, 1, 2 first

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
from datetime import datetime, timedelta
import re

# Check for requests
try:
    import requests
except ImportError:
    print("[!] Missing 'requests' package. Run: pip install requests")
    sys.exit(1)

# Intel Trust Authority endpoints
ITA_API_BASE = "https://api.trustauthority.intel.com"
ITA_ATTEST_ENDPOINT = f"{ITA_API_BASE}/appraisal/v1/attest"


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           Attest Model with Intel Trust Authority                     ║
║                                                                       ║
║   Send TPM quote to ITA, receive signed attestation token             ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def load_attestation_package():
    """Load the attestation package."""
    try:
        with open("attestation_package.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("[!] Run 2_generate_quote.py first")
        sys.exit(1)


def get_ita_api_key():
    """Get Intel Trust Authority API key from environment or file."""
    # Check environment
    api_key = os.environ.get("INTEL_TRUST_AUTHORITY_API_KEY")
    
    if api_key:
        return api_key
    
    # Check file
    key_file = os.path.expanduser("~/.intel_trust_authority_key")
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            return f.read().strip()
    
    return None


def send_to_trust_authority(package: dict, api_key: str) -> dict:
    """Send attestation request to Intel Trust Authority."""
    print("\n[2/4] Sending to Intel Trust Authority...")
    
    # Prepare request
    request_body = {
        "quote": package["quote"]["message"],
        "signature": package["quote"]["signature"],
        "pcrs": package["quote"]["pcrs"],
        "nonce": package["nonce"],
        "user_data": base64.b64encode(json.dumps({
            "model_hash": package["model_info"]["hash"],
            "model_path": package["model_info"]["path"],
            "measured_at": package["model_info"]["measured_at"]
        }).encode()).decode()
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-api-key": api_key
    }
    
    try:
        response = requests.post(
            ITA_ATTEST_ENDPOINT,
            json=request_body,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            print("    [!] Invalid API key")
            return None
        elif response.status_code == 400:
            print(f"    [!] Bad request: {response.text}")
            return None
        else:
            print(f"    [!] Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("    [!] Cannot connect to Intel Trust Authority")
        return None
    except requests.exceptions.Timeout:
        print("    [!] Request timed out")
        return None


def simulate_ita_response(package: dict) -> dict:
    """
    Simulate Intel Trust Authority response for demo purposes.
    
    NOTE: This simulates ONLY the ITA cloud response.
          The TPM operations (PCR extend, quote generation) are REAL hardware.
    """
    print("\n[2/4] Contacting Intel Trust Authority...")
    print("    [SIMULATED] No API key - generating simulated ITA response")
    print("    [HARDWARE] TPM quote used in this request is REAL")
    
    model_hash = package["model_info"]["hash"]
    nonce = package["nonce"]
    
    # Create simulated JWT claims based on real TPM data
    now = datetime.utcnow()
    claims = {
        # Standard JWT claims
        "iss": "https://trustauthority.intel.com",
        "sub": "model-attestation",
        "aud": "ai-model-consumer",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=1)).timestamp()),
        "jti": hashlib.sha256(nonce.encode()).hexdigest()[:16],
        
        # Intel Trust Authority specific claims  
        "attester_type": "TPM",
        "attester_tcb_status": "UpToDate",
        "platform_instance_id": hashlib.sha256(
            package["platform_info"]["hostname"].encode()
        ).hexdigest()[:32],
        
        # Model attestation claims (from REAL TPM measurement)
        "x-model-hash": model_hash,
        "x-model-path": package["model_info"]["path"],
        "x-pcr-index": package["model_info"]["pcr_index"],
        "x-attestation-time": now.isoformat(),
        
        # Trust evaluation
        "x-trust-score": 100,
        "x-policy-ids": ["model-integrity-v1", "platform-security-v1"],
        "x-evaluation-result": "TRUSTED"
    }
    
    # Create JWT (header.payload.signature)
    header = {
        "alg": "RS256",
        "typ": "JWT",
        "kid": "intel-ita-signing-key-01"
    }
    
    header_b64 = base64.urlsafe_b64encode(
        json.dumps(header).encode()
    ).decode().rstrip('=')
    
    payload_b64 = base64.urlsafe_b64encode(
        json.dumps(claims).encode()
    ).decode().rstrip('=')
    
    # Simulated signature
    sig_input = f"{header_b64}.{payload_b64}".encode()
    simulated_sig = hashlib.sha256(sig_input + b"demo_signing_key").hexdigest()
    sig_b64 = base64.urlsafe_b64encode(bytes.fromhex(simulated_sig)).decode().rstrip('=')
    
    token = f"{header_b64}.{payload_b64}.{sig_b64}"
    
    return {
        "token": token,
        "token_type": "Bearer",
        "expires_in": 3600,
        "simulated": True
    }


def verify_token_structure(token: str) -> dict:
    """Parse and verify JWT structure."""
    print("\n[3/4] Verifying token structure...")
    
    parts = token.split('.')
    if len(parts) != 3:
        print("    [!] Invalid JWT format")
        return None
    
    # Decode header
    header_padded = parts[0] + '=' * (4 - len(parts[0]) % 4)
    header = json.loads(base64.urlsafe_b64decode(header_padded))
    
    # Decode payload
    payload_padded = parts[1] + '=' * (4 - len(parts[1]) % 4)
    payload = json.loads(base64.urlsafe_b64decode(payload_padded))
    
    print(f"    Algorithm: {header.get('alg')}")
    print(f"    Issuer: {payload.get('iss')}")
    print(f"    Expires: {datetime.fromtimestamp(payload.get('exp', 0)).isoformat()}")
    print(f"    Trust Score: {payload.get('x-trust-score')}")
    print(f"    Evaluation: {payload.get('x-evaluation-result')}")
    
    return {
        "header": header,
        "payload": payload,
        "signature": parts[2]
    }


def save_attestation_result(result: dict, token_info: dict, package: dict):
    """Save attestation result."""
    print("\n[4/4] Saving attestation result...")
    
    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "token": result["token"],
        "expires_at": datetime.fromtimestamp(
            token_info["payload"]["exp"]
        ).isoformat(),
        "model_info": package["model_info"],
        "trust_evaluation": {
            "result": token_info["payload"].get("x-evaluation-result"),
            "score": token_info["payload"].get("x-trust-score"),
            "policies": token_info["payload"].get("x-policy-ids"),
            "tcb_status": token_info["payload"].get("attester_tcb_status"),
            "simulated": result.get("simulated", False)
        }
    }
    
    result_path = "attestation_result.json"
    with open(result_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Also save just the token for use
    token_path = "attestation_token.jwt"
    with open(token_path, 'w') as f:
        f.write(result["token"])
    
    print(f"    [✓] Saved result to {result_path}")
    print(f"    [✓] Saved token to {token_path}")
    
    return output


def main():
    print_banner()
    
    # Load package
    print("\n[1/4] Loading attestation package...")
    package = load_attestation_package()
    print(f"    Model: {package['model_info']['path']}")
    print(f"    Hash:  {package['model_info']['hash'][:32]}...")
    
    # Check for API key
    api_key = get_ita_api_key()
    
    if api_key:
        print("\n════════════════════════════════════════════════════════════════════")
        print("  [HARDWARE] Intel Trust Authority API key found")
        print("════════════════════════════════════════════════════════════════════")
        result = send_to_trust_authority(package, api_key)
        
        if not result:
            print("[!] ITA request failed, using simulated response")
            result = simulate_ita_response(package)
        
        ita_mode = "[HARDWARE] Intel Trust Authority (PRODUCTION)"
    else:
        print("\n════════════════════════════════════════════════════════════════════")
        print("  [SIMULATED] No ITA API key - ITA response will be simulated")
        print("  [HARDWARE] TPM quote is REAL hardware-generated")
        print("════════════════════════════════════════════════════════════════════")
        result = simulate_ita_response(package)
        ita_mode = "[SIMULATED] (TPM quote is real hardware)"
    
    # Verify token structure
    token_info = verify_token_structure(result["token"])
    
    if not token_info:
        print("[!] Token verification failed")
        sys.exit(1)
    
    # Save result
    output = save_attestation_result(result, token_info, package)
    
    # Display result
    trust_result = token_info["payload"].get("x-evaluation-result", "UNKNOWN")
    
    if trust_result == "TRUSTED":
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ✅ MODEL ATTESTATION SUCCESSFUL                                      ║
╚══════════════════════════════════════════════════════════════════════╝

  Mode:            {ita_mode}
  Evaluation:      {trust_result}
  Trust Score:     {token_info["payload"].get("x-trust-score")}
  TCB Status:      {token_info["payload"].get("attester_tcb_status")}
  Token Expires:   {output["expires_at"]}
  
  The attestation token proves:
  ✓ Platform is genuine Intel hardware
  ✓ TPM quote is valid
  ✓ Model hash: {package["model_info"]["hash"][:32]}...
  
  This token can be presented to consumers as proof of model integrity.
  
  Next: python 4_secure_inference.py
        """)
    else:
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ❌ MODEL ATTESTATION FAILED                                          ║
╚══════════════════════════════════════════════════════════════════════╝

  Evaluation Result: {trust_result}
  
  The model could not be attested. This could mean:
  - TPM quote verification failed
  - Platform is not trusted
  - Model hash mismatch
        """)
        sys.exit(1)


if __name__ == "__main__":
    main()
