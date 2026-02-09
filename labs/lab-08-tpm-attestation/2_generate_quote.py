#!/usr/bin/env python3
"""
Step 2: Generate TPM Quote

Creates a signed attestation quote from the TPM. The quote contains:
- Selected PCR values (including our model measurement)
- Nonce (freshness guarantee)
- TPM signature (AIK - Attestation Identity Key)

The quote proves the platform state at a specific moment.

Setup Requirements:
    1. sudo apt install tpm2-tools
    2. sudo usermod -aG tss $USER && newgrp tss
    3. Run 0_check_tpm.py and 1_measure_model.py first

Author: GopeshK
License: MIT License
Disclaimer: Educational purposes only.
"""

import os
import sys
import json
import base64
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
import secrets


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║               Generate TPM Quote for Attestation                      ║
║                                                                       ║
║   Creates signed evidence of platform and model state                 ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def load_tpm_status():
    """Load TPM availability status."""
    try:
        with open(".tpm_status.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("[!] Run 0_check_tpm.py first")
        sys.exit(1)


def load_measurement_record():
    """Load model measurement record."""
    try:
        with open("model_measurement.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("[!] Run 1_measure_model.py first")
        sys.exit(1)


def generate_nonce() -> bytes:
    """Generate a cryptographic nonce for freshness."""
    return secrets.token_bytes(32)


def read_all_pcrs() -> dict:
    """Read all PCR values."""
    result = subprocess.run(
        ["tpm2_pcrread", "sha256:all"],
        capture_output=True,
        text=True
    )
    
    pcr_values = {}
    
    if result.returncode == 0:
        current_pcr = None
        for line in result.stdout.split('\n'):
            line = line.strip()
            if 'sha256:' in line.lower():
                continue
            if ':' in line and '0x' in line:
                parts = line.split(':')
                try:
                    pcr_idx = int(parts[0].strip())
                    pcr_val = parts[1].strip()
                    if pcr_val.startswith('0x'):
                        pcr_val = pcr_val[2:]
                    pcr_values[pcr_idx] = pcr_val
                except (ValueError, IndexError):
                    pass
    
    return pcr_values


def create_attestation_key():
    """Create or load Attestation Identity Key (AIK)."""
    print("\n[1/4] Setting up Attestation Identity Key...")
    print("    [HARDWARE] Creating AIK in TPM")
    
    aik_pub = "aik.pub"
    aik_priv = "aik.priv"
    aik_ctx = "aik.ctx"
    
    # Check if AIK already exists
    if os.path.exists(aik_ctx):
        print("    [HARDWARE] ✓ Using existing AIK")
        return aik_ctx
    
    # Create primary key (Endorsement Hierarchy)
    result = subprocess.run(
        ["tpm2_createprimary", "-C", "e", "-g", "sha256", "-G", "rsa2048", "-c", "primary.ctx"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"    [!] Failed to create primary: {result.stderr}")
        return None
    
    # Create AIK with explicit signing scheme (rsassa for restricted signing)
    # The scheme must be specified for restricted signing keys
    result = subprocess.run(
        ["tpm2_create", "-C", "primary.ctx", "-g", "sha256", "-G", "rsa2048:rsassa:null",
         "-u", aik_pub, "-r", aik_priv, 
         "-a", "restricted|sign|fixedtpm|fixedparent|sensitivedataorigin|userwithauth"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"    [ERROR] Failed to create AIK: {result.stderr}")
        return None
    
    # Load AIK
    result = subprocess.run(
        ["tpm2_load", "-C", "primary.ctx", "-u", aik_pub, "-r", aik_priv, "-c", aik_ctx],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"    [ERROR] Failed to load AIK: {result.stderr}")
        return None
    
    print("    [HARDWARE] ✓ AIK created and loaded")
    
    # Cleanup intermediate files
    for f in ["primary.ctx"]:
        if os.path.exists(f):
            os.remove(f)
    
    return aik_ctx


def generate_tpm_quote(aik_ctx: str, nonce: bytes, pcr_selection: str) -> dict:
    """
    Generate a TPM quote - a signed snapshot of PCR values.
    
    The quote contains: PCR values + nonce + TPM signature (using AIK).
    This proves the platform state at this exact moment and cannot be forged.
    """
    print("\n[2/4] Generating TPM Quote...")
    print("    [HARDWARE] Calling tpm2_quote with AIK")
    
    # Write nonce to file
    nonce_file = "/tmp/quote_nonce.bin"
    with open(nonce_file, 'wb') as f:
        f.write(nonce)
    
    # Output files from tpm2_quote command:
    #
    # - quote.msg: The attestation structure containing PCR digest + nonce + metadata
    #              This is what gets signed - proves "these PCRs had these values at this time"
    #              Example content (binary, base64 encoded for transport):
    #              {"pcr_digest": "a3f2...89bc", "nonce": "user_provided_random", "clock": 12345}
    #
    # - quote.sig: TPM's RSA signature over quote.msg using the AIK private key
    #              Only real TPM hardware can produce this - proves authenticity
    #              Example: 256-byte RSA-2048 signature (base64: "MEUCIQD3x7...")
    #              Verifier uses AIK public key to validate: verify(quote.msg, quote.sig, aik.pub)
    #
    # - quote.pcrs: Raw PCR values that were included in the quote
    #              Verifier compares these against expected "golden" values
    #              Example for PCR 14:
    #              PCR[14] = sha256("previous_value" || sha256(model_weights.h5))
    #                      = 0x8a3b4c5d...  (expected hash if model is untampered)
    #
    quote_msg = "quote.msg"
    quote_sig = "quote.sig"
    quote_pcrs = "quote.pcrs"
    
    # tpm2_quote: Signs PCR values with AIK, includes nonce to prevent replay attacks
    # -c: AIK context, -l: PCR selection, -q: nonce file, -m/-s/-o: output files
    result = subprocess.run(
        ["tpm2_quote", "-c", aik_ctx, "-l", pcr_selection, 
         "-q", nonce_file, "-m", quote_msg, "-s", quote_sig, "-o", quote_pcrs],
        capture_output=True,
        text=True
    )
    
    os.remove(nonce_file)
    
    if result.returncode != 0:
        print(f"    [ERROR] Quote generation failed: {result.stderr}")
        return None
    
    # Read quote components
    quote_data = {}
    
    if os.path.exists(quote_msg):
        with open(quote_msg, 'rb') as f:
            quote_data["message"] = base64.b64encode(f.read()).decode()
    
    if os.path.exists(quote_sig):
        with open(quote_sig, 'rb') as f:
            quote_data["signature"] = base64.b64encode(f.read()).decode()
    
    if os.path.exists(quote_pcrs):
        with open(quote_pcrs, 'rb') as f:
            quote_data["pcrs"] = base64.b64encode(f.read()).decode()
    
    print("    [HARDWARE] ✓ Quote generated and signed by TPM")
    print(f"    [DEBUG] quote_data: {json.dumps(quote_data, indent=2)}")
    
    return quote_data


def save_quote_package(nonce: bytes, quote_data: dict, measurement: dict):
    """Save complete quote package for attestation."""
    print("\n[3/4] Creating attestation package...")
    
    package = {
        "timestamp": datetime.utcnow().isoformat(),
        "nonce": base64.b64encode(nonce).decode(),
        "quote": quote_data,
        "model_info": {
            "path": measurement.get("model_path"),
            "hash": measurement.get("model_hash"),
            "pcr_index": measurement.get("pcr_index"),
            "measured_at": measurement.get("timestamp")
        },
        "platform_info": {
            "hostname": os.uname().nodename,
            "kernel": os.uname().release,
            "architecture": os.uname().machine
        }
    }
    
    package_path = "attestation_package.json"
    with open(package_path, 'w') as f:
        json.dump(package, f, indent=2)
    
    print(f"    [✓] Saved to {package_path}")
    print(f"    [LOG] attestation_package.json contents:")
    print(json.dumps(package, indent=2))
    
    return package


def display_quote_info(package: dict):
    """Display quote information."""
    print("\n[4/4] Quote Summary:")
    
    model = package.get("model_info", {})
    
    print(f"""
    Mode:          HARDWARE TPM
    Nonce:         {package.get('nonce', 'N/A')[:32]}...
    Model Hash:    {model.get('hash', 'N/A')[:32]}...
    PCR Index:     {model.get('pcr_index', 'N/A')}
    Timestamp:     {package.get('timestamp', 'N/A')}
    """)


def main():
    print_banner()
    
    # Load status and measurement
    status = load_tpm_status()
    measurement = load_measurement_record()
    
    if not status.get("tpm_available", False):
        print("❌ TPM 2.0 hardware not available!")
        print("   This lab requires real Intel TPM hardware.")
        sys.exit(1)
    
    print("════════════════════════════════════════════════════════════════════")
    print("  [HARDWARE] TPM 2.0 available - generating hardware quote")
    print("════════════════════════════════════════════════════════════════════")
    
    # Generate nonce
    print("\n[0/4] Generating nonce for freshness...")
    nonce = generate_nonce()
    print(f"    Nonce: {nonce.hex()[:32]}...")
    
    # Get PCR index from step 1's measurement record
    # Default is 14 (first application-use PCR per TCG spec: 0-13 reserved for boot chain)
    pcr_index = measurement.get("pcr_index", 14)
    
    # Create AIK
    aik_ctx = create_attestation_key()
    
    if not aik_ctx:
        print("❌ AIK creation failed!")
        print("   Check TPM permissions and tpm2-tools installation.")
        sys.exit(1)
    
    # Generate real quote
    quote_data = generate_tpm_quote(aik_ctx, nonce, f"sha256:{pcr_index}")
    
    if not quote_data:
        print("❌ Quote generation failed!")
        print("   Check TPM hardware and permissions.")
        sys.exit(1)
    
    # Save package
    package = save_quote_package(nonce, quote_data, measurement)
    
    # Display info
    display_quote_info(package)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ✅ TPM QUOTE GENERATED                                               ║
╚══════════════════════════════════════════════════════════════════════╝

  The attestation package contains:
  - TPM quote (signed PCR values)
  - Nonce (prevents replay attacks)
  - Model measurement info
  - Platform metadata
  
  This package can be sent to Intel Trust Authority for verification.
  
  Next: python 3_attest_model.py
    """)


if __name__ == "__main__":
    main()
