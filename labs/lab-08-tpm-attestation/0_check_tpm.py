#!/usr/bin/env python3
"""
Step 0: Check TPM 2.0 Hardware Availability

Verifies that Intel/AMD TPM 2.0 is available for hardware-rooted attestation.
TPM provides unforgeable cryptographic identity tied to the physical hardware.

Setup Requirements:
    1. sudo apt install tpm2-tools
    2. sudo usermod -aG tss $USER && newgrp tss
    3. (Optional) export INTEL_TRUST_AUTHORITY_API_KEY="your-key"

Author: GopeshK
License: MIT License
Disclaimer: Educational purposes only.
"""

import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║       TPM 2.0 Hardware Check for AI Model Attestation                 ║
║                                                                       ║
║   TPM = Trusted Platform Module (Hardware Security Chip)              ║
║   Provides: Unforgeable identity, secure key storage, attestation     ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


def check_tpm_device():
    """Check if TPM device exists."""
    print("\n[1/5] Checking TPM Device...")
    
    tpm_paths = ["/dev/tpm0", "/dev/tpmrm0"]
    found = []
    
    for path in tpm_paths:
        if os.path.exists(path):
            found.append(path)
            print(f"    [✓] Found: {path}")
        else:
            print(f"    [!] Not found: {path}")
    
    if not found:
        print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  ❌ TPM DEVICE NOT FOUND                                        ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Possible solutions:
    1. Enable TPM in BIOS (Security settings)
    2. Load the TPM kernel module: sudo modprobe tpm_tis
    3. Check if running in a VM (VMs may not expose TPM)
    4. Use a cloud instance with vTPM (Azure, GCP, AWS Nitro)
        """)
        return False
    
    return True


def check_tpm_version():
    """Check TPM version (must be 2.0)."""
    print("\n[2/5] Checking TPM Version...")
    
    version_path = "/sys/class/tpm/tpm0/tpm_version_major"
    
    try:
        with open(version_path, 'r') as f:
            version = f.read().strip()
        
        if version == "2":
            print(f"    [✓] TPM Version: 2.0 (required for modern attestation)")
            return True
        else:
            print(f"    [!] TPM Version: {version}.x (version 2.0 required)")
            return False
    except FileNotFoundError:
        print("    [!] Could not determine TPM version")
        return False


def check_tpm_permissions():
    """Check if user has access to TPM device."""
    print("\n[3/5] Checking TPM Permissions...")
    
    # Check /dev/tpmrm0 (Resource Manager - preferred for multi-process access)
    tpmrm_path = "/dev/tpmrm0"
    
    if os.path.exists(tpmrm_path):
        # Check if readable
        if os.access(tpmrm_path, os.R_OK | os.W_OK):
            print(f"    [✓] User has read/write access to {tpmrm_path}")
            return True
        else:
            import grp
            import pwd
            
            stat_info = os.stat(tpmrm_path)
            owner_group = grp.getgrgid(stat_info.st_gid).gr_name
            current_user = pwd.getpwuid(os.getuid()).pw_name
            
            print(f"    [!] Permission denied on {tpmrm_path}")
            print(f"        Device owned by group: {owner_group}")
            print(f"        Current user: {current_user}")
            print(f"")
            print(f"    Fix: sudo usermod -aG {owner_group} {current_user}")
            print(f"         Then log out and log back in")
            return False
    
    return False


def check_tpm_tools():
    """Check if tpm2-tools are installed."""
    print("\n[4/5] Checking TPM Tools...")
    
    tools = ["tpm2_getrandom", "tpm2_pcrread", "tpm2_createek", "tpm2_quote"]
    missing = []
    
    for tool in tools:
        result = subprocess.run(["which", tool], capture_output=True)
        if result.returncode == 0:
            print(f"    [✓] {tool}: installed")
        else:
            print(f"    [!] {tool}: NOT found")
            missing.append(tool)
    
    if missing:
        print("""
    Install tpm2-tools:
        Ubuntu/Debian: sudo apt install tpm2-tools
        Fedora/RHEL:   sudo dnf install tpm2-tools
        """)
        return False
    
    return True


def check_tpm_functionality():
    """Test basic TPM functionality."""
    print("\n[5/5] Testing TPM Functionality...")
    
    # Try to get random bytes from TPM
    try:
        result = subprocess.run(
            ["tpm2_getrandom", "--hex", "8"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            random_hex = result.stdout.strip()
            print(f"    [✓] TPM random generation works")
            print(f"        Sample: 0x{random_hex}")
        else:
            print(f"    [!] TPM random generation failed")
            print(f"        Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("    [!] TPM command timed out")
        return False
    except FileNotFoundError:
        print("    [!] tpm2_getrandom not found")
        return False
    
    # Read PCR values
    try:
        result = subprocess.run(
            ["tpm2_pcrread", "sha256:0,1,7"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print(f"    [✓] PCR reading works")
            # Show first PCR value
            lines = result.stdout.strip().split('\n')
            for line in lines[:3]:
                if 'sha256' in line.lower() or '0x' in line:
                    print(f"        {line.strip()}")
        else:
            print(f"    [!] PCR reading failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"    [!] PCR reading error: {e}")
        return False
    
    return True


def print_summary(results):
    """Print final summary."""
    all_pass = all(results.values())
    
    print("\n" + "="*70)
    print(" TPM HARDWARE CHECK SUMMARY ".center(70, "="))
    print("="*70)
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check:<30} {status}")
    
    print("="*70)
    
    if all_pass:
        print("""
    ════════════════════════════════════════════════════════════════════
     [HARDWARE] TPM 2.0 READY FOR AI MODEL ATTESTATION
    ════════════════════════════════════════════════════════════════════
    
    Your system has hardware TPM 2.0 available for:
    * Unforgeable platform identity (Endorsement Key)
    * Measured boot chain (PCR registers)
    * Model hash binding (extend PCR with model digest)
    * Remote attestation (Intel Trust Authority integration)
    
    Next: python 1_measure_model.py
        """)
    else:
        print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  ❌ TPM HARDWARE REQUIRED                                        ║
    ╚════════════════════════════════════════════════════════════════╝
    
    This lab requires Intel TPM 2.0 hardware.
    No simulation mode is available - we demonstrate REAL attestation.
    
    To resolve:
    1. Ensure you're on Intel hardware with TPM 2.0
    2. Check BIOS to enable TPM
    3. Install tpm2-tools: sudo apt install tpm2-tools
    4. Add user to tss group: sudo usermod -aG tss $USER
        """)
        sys.exit(1)
    
    return all_pass


def main():
    print_banner()
    
    results = {
        "TPM Device": check_tpm_device(),
        "TPM Version 2.0": check_tpm_version(),
        "TPM Permissions": check_tpm_permissions(),
        "TPM Tools": check_tpm_tools(),
        "TPM Functional": check_tpm_functionality(),
    }
    
    tpm_ready = print_summary(results)
    
    # Save status for other scripts
    import json
    status = {
        "tpm_available": tpm_ready,
        "checks": {k: v for k, v in results.items()}
    }
    
    with open(".tpm_status.json", "w") as f:
        json.dump(status, f, indent=2)
    
    print("[i] Status saved to .tpm_status.json")


if __name__ == "__main__":
    main()
