#!/usr/bin/env python3
"""
Memory Encryption Demonstration: TDX vs Standard VM

This script demonstrates the difference between:
- TDX VM: Memory is hardware-encrypted (data is protected)
- Standard VM: Memory is plaintext (data is exposed)

Run this on BOTH VMs to see the security difference.

Author: GopeshK
License: MIT
"""

import os
import sys
import time
import random
import subprocess
import json
import hashlib
from datetime import datetime

# ============================================================================
# Terminal Colors
# ============================================================================

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'

C = Colors

# ============================================================================
# Helper Functions
# ============================================================================

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def type_effect(text, delay=0.02):
    """Type text with delay effect"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def run_cmd(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() or result.stderr.strip()
    except Exception as e:
        return ""

def print_box(title, content, color=C.CYAN):
    """Print content in a box"""
    lines = content.split('\n')
    width = max(len(line) for line in lines + [title]) + 4
    
    print(f"{color}┌{'─' * width}┐{C.RESET}")
    print(f"{color}│{C.RESET} {C.BOLD}{title}{C.RESET}{' ' * (width - len(title) - 1)}{color}│{C.RESET}")
    print(f"{color}├{'─' * width}┤{C.RESET}")
    for line in lines:
        padding = width - len(line) - 1
        print(f"{color}│{C.RESET} {line}{' ' * padding}{color}│{C.RESET}")
    print(f"{color}└{'─' * width}┘{C.RESET}")

# ============================================================================
# TDX Detection
# ============================================================================

def detect_tdx_status():
    """Comprehensive TDX detection"""
    results = {
        "tdx_guest": False,
        "memory_encrypted": False,
        "attestation_device": False,
        "cpu_flags": False,
        "details": {}
    }
    
    # Check 1: TDX guest mode from dmesg
    dmesg_output = run_cmd("sudo dmesg 2>/dev/null | grep -i 'tdx'")
    if "Guest detected" in dmesg_output or "tdx_guest" in dmesg_output.lower():
        results["tdx_guest"] = True
        results["details"]["dmesg"] = dmesg_output.split('\n')[0]
    
    # Check 2: Memory encryption
    mem_encrypt = run_cmd("sudo dmesg 2>/dev/null | grep -i 'Memory Encryption'")
    if "Intel TDX" in mem_encrypt:
        results["memory_encrypted"] = True
        results["details"]["encryption"] = "Intel TDX AES-256-XTS"
    
    # Check 3: Attestation device
    if os.path.exists("/dev/tdx_guest"):
        results["attestation_device"] = True
        results["details"]["attestation"] = "/dev/tdx_guest present"
    
    # Check 4: CPU flags
    cpuinfo = run_cmd("grep -o 'tdx[_a-z]*' /proc/cpuinfo 2>/dev/null | head -1")
    if cpuinfo:
        results["cpu_flags"] = True
        results["details"]["cpu_flags"] = cpuinfo
    
    # Determine overall status
    results["is_tdx"] = results["tdx_guest"] or results["memory_encrypted"] or results["attestation_device"]
    
    return results

# ============================================================================
# Simulated Data
# ============================================================================

def generate_sensitive_data():
    """Generate simulated sensitive AI model data"""
    return {
        "model_name": "ProprietaryLLM-v3.7",
        "company": "AcmeCorp",
        "training_cost": "$15,000,000",
        "model_weights": [random.uniform(-1, 1) for _ in range(20)],
        "api_key": "sk-" + hashlib.sha256(str(random.random()).encode()).hexdigest()[:32],
        "customer_data": [
            {"user": "john.doe@example.com", "ssn": "123-45-6789"},
            {"user": "jane.smith@example.com", "ssn": "987-65-4321"},
        ],
        "inference_prompt": "Patient diagnosis: diabetes type 2, recommended treatment...",
    }

def generate_encrypted_garbage(length=64):
    """Generate random bytes that look like encrypted data"""
    return bytes([random.randint(0, 255) for _ in range(length)])

# ============================================================================
# Demo Scenarios
# ============================================================================

def show_intro():
    """Show introduction"""
    clear_screen()
    print(f"""
{C.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║   {C.BOLD}MEMORY ENCRYPTION DEMONSTRATION{C.RESET}{C.CYAN}                                            ║
║   {C.WHITE}Intel TDX vs Standard VM{C.CYAN}                                                      ║
║                                                                                ║
║   This demo shows what a malicious hypervisor/cloud operator would see        ║
║   when attempting to read VM memory containing sensitive AI data.             ║
║                                                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝{C.RESET}
""")
    time.sleep(1)

def show_vm_status(tdx_status):
    """Display current VM security status"""
    
    if tdx_status["is_tdx"]:
        status_color = C.GREEN
        status_text = "TDX PROTECTED"
        status_icon = "🔒"
        memory_status = "ENCRYPTED (AES-256)"
        hypervisor_access = "BLOCKED"
    else:
        status_color = C.RED
        status_text = "STANDARD VM (UNPROTECTED)"
        status_icon = "⚠️"
        memory_status = "PLAINTEXT"
        hypervisor_access = "FULL ACCESS"
    
    print(f"\n{C.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}")
    print(f"  {C.BOLD}VM SECURITY STATUS{C.RESET}")
    print(f"{C.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}")
    print()
    print(f"  {status_icon} Status:           {status_color}{C.BOLD}{status_text}{C.RESET}")
    print(f"  📍 Memory State:      {status_color}{memory_status}{C.RESET}")
    print(f"  🔑 Hypervisor Access: {status_color}{hypervisor_access}{C.RESET}")
    
    if tdx_status["is_tdx"]:
        print(f"  ✅ TDX Guest Mode:    {C.GREEN}Active{C.RESET}")
        print(f"  ✅ Attestation:       {C.GREEN}Available{C.RESET}")
    else:
        print(f"  ❌ TDX Guest Mode:    {C.RED}Not Active{C.RESET}")
        print(f"  ❌ Attestation:       {C.RED}Not Available{C.RESET}")
    
    print()
    return tdx_status["is_tdx"]

def show_attack_scenario():
    """Explain the attack scenario"""
    print(f"\n{C.YELLOW}{'═' * 78}{C.RESET}")
    print(f"{C.YELLOW}  SCENARIO: Malicious Cloud Provider Attack{C.RESET}")
    print(f"{C.YELLOW}{'═' * 78}{C.RESET}")
    print(f"""
  {C.BOLD}THREAT MODEL:{C.RESET}
  ─────────────
  Your company has deployed a proprietary AI model worth $15M to a cloud
  provider. A malicious administrator wants to steal the model and data.

  {C.BOLD}ATTACK VECTOR:{C.RESET}
  ─────────────
  The attacker has hypervisor-level access and attempts to:
  1. Read physical memory pages of your VM
  2. Extract model weights and architecture
  3. Steal customer data being processed
  4. Capture API keys and secrets

  {C.BOLD}WHAT HAPPENS NEXT:{C.RESET}
  ─────────────────
  The outcome depends on whether TDX memory encryption is enabled...
""")
    input(f"\n  {C.CYAN}Press Enter to simulate the attack...{C.RESET}")

def simulate_attack(is_tdx_protected):
    """Simulate hypervisor memory reading attack"""
    
    sensitive_data = generate_sensitive_data()
    
    print(f"\n{C.BLUE}[HYPERVISOR]{C.RESET} Initiating memory scan...")
    time.sleep(0.5)
    
    print(f"{C.BLUE}[HYPERVISOR]{C.RESET} Locating VM memory pages...")
    time.sleep(0.5)
    
    print(f"{C.BLUE}[HYPERVISOR]{C.RESET} Reading physical memory at 0x7f8a3c000000...")
    time.sleep(0.5)
    
    print(f"\n{'─' * 78}")
    
    if is_tdx_protected:
        # TDX Protected - Show encrypted garbage
        print(f"{C.GREEN}{C.BOLD}  MEMORY CONTENTS (TDX ENCRYPTED){C.RESET}")
        print(f"{'─' * 78}")
        print()
        
        # Show encrypted bytes (what hypervisor actually sees)
        for i in range(6):
            encrypted_bytes = generate_encrypted_garbage(32)
            hex_line = ' '.join(f'{b:02x}' for b in encrypted_bytes)
            ascii_repr = ''.join(chr(b) if 32 <= b < 127 else '.' for b in encrypted_bytes[:16])
            print(f"  {C.DIM}{hex_line}  |{ascii_repr}|{C.RESET}")
            time.sleep(0.1)
        
        print()
        print(f"{'─' * 78}")
        print(f"""
  {C.GREEN}╔════════════════════════════════════════════════════════════════════════╗
  ║  🔒 TDX PROTECTION ACTIVE - ATTACK FAILED!                             ║
  ╠════════════════════════════════════════════════════════════════════════╣
  ║                                                                        ║
  ║  {C.BOLD}What the attacker sees:{C.RESET}{C.GREEN}                                              ║
  ║  • Random encrypted bytes (AES-256-XTS)                                ║
  ║  • No readable strings or patterns                                     ║
  ║  • Cannot extract model weights                                        ║
  ║  • Cannot find API keys or secrets                                     ║
  ║  • Customer data is completely protected                               ║
  ║                                                                        ║
  ║  {C.BOLD}Why it's protected:{C.RESET}{C.GREEN}                                                  ║
  ║  • Memory encrypted by Intel TDX hardware on CPU                       ║
  ║  • Encryption keys never leave the CPU package                         ║
  ║  • Hypervisor has NO access to decryption keys                         ║
  ║  • Hardware enforced - cannot be bypassed by software                  ║
  ║                                                                        ║
  ╚════════════════════════════════════════════════════════════════════════╝{C.RESET}
""")
        
    else:
        # Standard VM - Show plaintext data (VULNERABLE)
        print(f"{C.RED}{C.BOLD}  MEMORY CONTENTS (PLAINTEXT - VULNERABLE!){C.RESET}")
        print(f"{'─' * 78}")
        print()
        
        # Show actual "stolen" data
        print(f"  {C.RED}[EXTRACTED] Model Name:{C.RESET} {sensitive_data['model_name']}")
        time.sleep(0.2)
        print(f"  {C.RED}[EXTRACTED] Company:{C.RESET} {sensitive_data['company']}")
        time.sleep(0.2)
        print(f"  {C.RED}[EXTRACTED] Training Cost:{C.RESET} {sensitive_data['training_cost']}")
        time.sleep(0.2)
        
        print(f"\n  {C.RED}{C.BOLD}[STOLEN] Model Weights (first 10):{C.RESET}")
        for i, w in enumerate(sensitive_data['model_weights'][:10]):
            print(f"    weight[{i}] = {w:.8f}")
            time.sleep(0.05)
        
        print(f"\n  {C.RED}{C.BOLD}[STOLEN] API Key:{C.RESET}")
        print(f"    {sensitive_data['api_key']}")
        time.sleep(0.2)
        
        print(f"\n  {C.RED}{C.BOLD}[STOLEN] Customer PII:{C.RESET}")
        for customer in sensitive_data['customer_data']:
            print(f"    Email: {customer['user']}, SSN: {customer['ssn']}")
            time.sleep(0.1)
        
        print(f"\n  {C.RED}{C.BOLD}[STOLEN] Inference Data:{C.RESET}")
        print(f"    {sensitive_data['inference_prompt']}")
        
        print()
        print(f"{'─' * 78}")
        print(f"""
  {C.RED}╔════════════════════════════════════════════════════════════════════════╗
  ║  ⚠️  NO PROTECTION - ATTACK SUCCESSFUL!                                ║
  ╠════════════════════════════════════════════════════════════════════════╣
  ║                                                                        ║
  ║  {C.BOLD}What the attacker stole:{C.RESET}{C.RED}                                             ║
  ║  ✗ Complete model architecture and weights                             ║
  ║  ✗ API keys and authentication tokens                                  ║
  ║  ✗ Customer personal data (emails, SSNs)                               ║
  ║  ✗ Inference requests and responses                                    ║
  ║  ✗ All secrets stored in memory                                        ║
  ║                                                                        ║
  ║  {C.BOLD}Why it's vulnerable:{C.RESET}{C.RED}                                                 ║
  ║  • Memory is stored in plaintext (no encryption)                       ║
  ║  • Hypervisor has direct access to VM physical memory                  ║
  ║  • Cloud operator can read all data at any time                        ║
  ║  • No hardware protection against privileged attackers                 ║
  ║                                                                        ║
  ║  {C.BOLD}FIX: Enable Intel TDX for hardware memory encryption!{C.RESET}{C.RED}                ║
  ║                                                                        ║
  ╚════════════════════════════════════════════════════════════════════════╝{C.RESET}
""")

def show_side_by_side_comparison():
    """Show side-by-side comparison"""
    print(f"\n{C.CYAN}{'═' * 78}{C.RESET}")
    print(f"{C.CYAN}  COMPARISON SUMMARY{C.RESET}")
    print(f"{C.CYAN}{'═' * 78}{C.RESET}")
    
    print(f"""
┌─────────────────────────────────────┬─────────────────────────────────────┐
│  {C.GREEN}{C.BOLD}TDX VM (This VM){C.RESET}                    │  {C.RED}{C.BOLD}Standard VM{C.RESET}                       │
├─────────────────────────────────────┼─────────────────────────────────────┤
│                                     │                                     │
│  Memory: {C.GREEN}████████████{C.RESET} ENCRYPTED    │  Memory: {C.RED}░░░░░░░░░░░░{C.RESET} PLAINTEXT   │
│                                     │                                     │
│  Hypervisor sees:                   │  Hypervisor sees:                   │
│  {C.DIM}a7 f3 2b 9c 4e 8d 1a b0...{C.RESET}       │  {C.WHITE}"model_weights": [0.234...]{C.RESET}     │
│  {C.DIM}(random encrypted bytes){C.RESET}         │  {C.WHITE}"api_key": "sk-abc123..."{C.RESET}       │
│                                     │                                     │
│  {C.GREEN}✅ Model weights: PROTECTED{C.RESET}       │  {C.RED}❌ Model weights: STOLEN{C.RESET}         │
│  {C.GREEN}✅ API keys: PROTECTED{C.RESET}            │  {C.RED}❌ API keys: STOLEN{C.RESET}              │
│  {C.GREEN}✅ Customer data: PROTECTED{C.RESET}       │  {C.RED}❌ Customer data: STOLEN{C.RESET}         │
│  {C.GREEN}✅ Inference data: PROTECTED{C.RESET}      │  {C.RED}❌ Inference data: STOLEN{C.RESET}        │
│                                     │                                     │
│  {C.GREEN}Hardware Root of Trust{C.RESET}            │  {C.RED}No Hardware Protection{C.RESET}           │
│                                     │                                     │
└─────────────────────────────────────┴─────────────────────────────────────┘
""")

def show_recommendations(is_tdx):
    """Show recommendations based on current VM status"""
    print(f"\n{C.CYAN}{'═' * 78}{C.RESET}")
    print(f"{C.CYAN}  RECOMMENDATIONS{C.RESET}")
    print(f"{C.CYAN}{'═' * 78}{C.RESET}")
    
    if is_tdx:
        print(f"""
  {C.GREEN}Your VM is protected by Intel TDX!{C.RESET}

  {C.BOLD}Best Practices:{C.RESET}
  ─────────────────
  1. {C.GREEN}✓{C.RESET} Use remote attestation to prove TDX status to clients
  2. {C.GREEN}✓{C.RESET} Implement secure boot chain
  3. {C.GREEN}✓{C.RESET} Keep VM software updated
  4. {C.GREEN}✓{C.RESET} Use TLS for data in transit
  5. {C.GREEN}✓{C.RESET} Monitor for abnormal access patterns

  {C.BOLD}Next Steps:{C.RESET}
  ───────────
  • Run: python3 4_tdx_attestation_demo.py  (to see attestation)
  • Run: python3 6_secure_ai_inference.py   (secure inference demo)
""")
    else:
        print(f"""
  {C.RED}⚠️  Your VM is NOT protected!{C.RESET}

  {C.BOLD}Immediate Actions:{C.RESET}
  ─────────────────────
  1. {C.RED}✗{C.RESET} Do NOT deploy sensitive AI models to this VM
  2. {C.RED}✗{C.RESET} Do NOT process customer PII
  3. {C.RED}✗{C.RESET} Do NOT store API keys or secrets

  {C.BOLD}To Enable Protection:{C.RESET}
  ─────────────────────
  Create a new VM with TDX enabled:

  {C.CYAN}gcloud compute instances create tdx-vm \\
    --machine-type=c3-standard-4 \\
    --confidential-compute-type=TDX \\
    --min-cpu-platform="Intel Sapphire Rapids" \\
    --zone=us-central1-a{C.RESET}

  {C.BOLD}Or use the deployment script:{C.RESET}
  ────────────────────────────────
  ./deploy_vms.sh --tdx-only
""")

# ============================================================================
# Main
# ============================================================================

def main():
    """Main demo function"""
    show_intro()
    
    # Detect TDX status
    print(f"  {C.CYAN}Detecting VM security status...{C.RESET}")
    time.sleep(1)
    tdx_status = detect_tdx_status()
    
    # Show current VM status
    is_tdx = show_vm_status(tdx_status)
    
    # Show attack scenario
    show_attack_scenario()
    
    # Simulate the attack
    simulate_attack(is_tdx)
    
    time.sleep(1)
    
    # Show comparison
    show_side_by_side_comparison()
    
    # Show recommendations
    show_recommendations(is_tdx)
    
    # Final summary
    print(f"\n{C.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}")
    if is_tdx:
        print(f"  {C.GREEN}{C.BOLD}✅ This VM is PROTECTED by Intel TDX hardware encryption{C.RESET}")
    else:
        print(f"  {C.RED}{C.BOLD}❌ This VM is VULNERABLE - Memory is plaintext{C.RESET}")
    print(f"{C.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}")
    print()
    
    # Return exit code
    return 0 if is_tdx else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Demo interrupted.{C.RESET}")
        sys.exit(130)
