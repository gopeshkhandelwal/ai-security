#!/usr/bin/env python3
"""
Memory Encryption Demonstration: TDX vs Standard VM

This script demonstrates the difference between:
- TDX VM: Memory is hardware-encrypted (data is protected)
- Standard VM: Memory is plaintext (data is exposed)

Usage:
  python3 2_memory_comparison_demo.py          # Auto-detect TDX status
  python3 2_memory_comparison_demo.py --demo   # Show BOTH scenarios for demo

The --demo flag shows both TDX and Standard VM scenarios side-by-side,
useful for presentations when you only have one VM available.

============================================================================
IMPORTANT TECHNICAL NOTE:
============================================================================
TDX encryption is TRANSPARENT to the guest OS. From within ANY VM (TDX or
not), memory always appears as plaintext because the CPU decrypts data before
delivering it to the process.

The encryption happens at the HARDWARE level:
  CPU Cache (plaintext) --> Memory Controller (AES-256 encrypt) --> RAM (encrypted)

To see REAL encrypted memory, you would need HYPERVISOR-LEVEL access to read
physical RAM - which cloud providers (Google, Azure, AWS) do not expose.

Therefore, this demo:
- Shows REAL plaintext memory reads (using ctypes) for Standard VM scenario
- Shows SIMULATED encrypted bytes for TDX scenario (what hypervisor would see)

This is an educational demonstration, not a security proof. For actual TDX
verification, use remote attestation (see 3_tdx_attestation_demo.py).
============================================================================

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
import struct
import ctypes
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
# Real Memory Operations
# ============================================================================

class SensitiveData:
    """
    Holds sensitive data in actual memory that we can inspect.
    This represents what would be in memory during AI inference.
    """
    def __init__(self):
        # These are REAL values stored in REAL memory
        self.model_name = "ProprietaryLLM-v3.7"
        self.company = "AcmeCorp"
        self.training_cost = "$15,000,000"
        self.api_key = "sk-LIVE-a]8Kp2mN9xR4vL7wQ1uF6tY3eH0jB5cZ"
        self.customer_ssn = "123-45-6789"
        self.customer_email = "john.doe@example.com"
        self.model_weights = [0.8234, -0.4521, 0.9182, -0.3345, 0.7721, 
                              -0.1923, 0.5567, -0.8891, 0.2234, -0.6678]
        self.inference_prompt = "Patient diagnosis: diabetes type 2"
        
        # Store memory addresses for demonstration
        self._addresses = {}
        self._store_addresses()
    
    def _store_addresses(self):
        """Get actual memory addresses of our data"""
        self._addresses['api_key'] = hex(id(self.api_key))
        self._addresses['customer_ssn'] = hex(id(self.customer_ssn))
        self._addresses['model_weights'] = hex(id(self.model_weights))

def read_memory_bytes(obj, num_bytes=64):
    """
    Read actual bytes from memory where an object is stored using ctypes.
    
    This reads REAL memory at the object's address. For Python objects, the
    bytes include the PyObject header (refcount, type pointer, etc.) followed
    by the actual data. This demonstrates that memory CAN be read - a hypervisor
    would see these same bytes (or plaintext string data at the right offset).
    """
    try:
        # Get the memory address of the object
        address = id(obj)
        
        # Create a ctypes pointer to read raw memory
        # Note: This reads the Python object header + data
        byte_array = (ctypes.c_char * num_bytes).from_address(address)
        return bytes(byte_array)
    except Exception as e:
        return None

def get_string_memory_content(s):
    """
    Read the actual bytes of a string from memory.
    Returns the raw memory representation.
    """
    try:
        # For Python strings, the actual character data is stored after the object header
        # We'll read the raw bytes that make up this string
        encoded = s.encode('utf-8')
        address = id(encoded)
        byte_array = (ctypes.c_char * len(encoded)).from_address(address + 32)  # Skip PyObject header
        return bytes(byte_array)
    except:
        return s.encode('utf-8')

def format_memory_dump(data_bytes, start_addr=0x7f8a3c000000):
    """Format bytes as a hex dump like a memory viewer"""
    lines = []
    for i in range(0, min(len(data_bytes), 96), 16):
        chunk = data_bytes[i:i+16]
        hex_part = ' '.join(f'{b:02x}' for b in chunk)
        ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        addr = start_addr + i
        lines.append(f"  {addr:016x}  {hex_part:<48}  |{ascii_part}|")
    return lines

def generate_encrypted_garbage(length=64):
    """
    Generate random bytes to SIMULATE what TDX-encrypted memory looks like.
    
    ============================================================================
    WHY THIS IS SIMULATED (NOT REAL ENCRYPTED DATA):
    ============================================================================
    TDX encryption is transparent to the guest OS. From inside any VM, we
    ALWAYS see plaintext memory because:
    
    1. The CPU decrypts data BEFORE delivering to the process
    2. Encryption happens at the hardware level (memory controller)
    3. Only the HYPERVISOR sees encrypted physical RAM pages
    
    To read REAL encrypted memory, we would need:
    - Hypervisor-level access (not available on cloud VMs)
    - Direct physical RAM access from the host machine
    
    This function generates random bytes that REPRESENT what the hypervisor
    would see when reading TDX-encrypted memory. AES-256-XTS encrypted data
    appears as uniformly random bytes with no discernible pattern.
    ============================================================================
    """
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
╠══════════════════════════════════════════════════════════════════════════════╣
║  {C.YELLOW}NOTE: TDX encryption is transparent to guests. From inside any VM,{C.CYAN}          ║
║  {C.YELLOW}memory appears as plaintext. Encrypted bytes shown for TDX scenario{C.CYAN}         ║
║  {C.YELLOW}are SIMULATED to represent what the hypervisor would see.{C.CYAN}                   ║
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

def simulate_attack(is_tdx_protected, sensitive_data=None):
    """
    Simulate hypervisor memory reading attack.
    
    ============================================================================
    WHAT THIS DEMO SHOWS:
    ============================================================================
    
    STANDARD VM (is_tdx_protected=False):
      - Uses ctypes to read REAL bytes from actual memory addresses
      - Shows actual plaintext data that hypervisor could steal
      - This is what really happens - no simulation
    
    TDX VM (is_tdx_protected=True):
      - Shows SIMULATED encrypted bytes (random data)
      - CANNOT read real encrypted RAM from within the VM
      - TDX encryption happens at hardware level, transparent to guest
      - The random bytes REPRESENT what hypervisor would see
    
    ============================================================================
    WHY WE CANNOT SHOW REAL TDX ENCRYPTED DATA:
    ============================================================================
    TDX encryption occurs in the memory controller hardware. From the VM's
    perspective (guest OS), memory is always decrypted by the CPU before
    being accessed. Only code running at the hypervisor level on the host
    machine can see the actual encrypted physical memory pages.
    
    Cloud providers (GCP, Azure, AWS) do not expose hypervisor-level access.
    ============================================================================
    """
    
    # Create REAL sensitive data in memory
    if sensitive_data is None:
        sensitive_data = SensitiveData()
    
    print(f"\n{C.BLUE}[HYPERVISOR]{C.RESET} Initiating memory scan...")
    time.sleep(0.5)
    
    # Show actual memory addresses
    print(f"{C.BLUE}[HYPERVISOR]{C.RESET} Found sensitive data at addresses:")
    print(f"    API Key:       {sensitive_data._addresses['api_key']}")
    print(f"    Customer SSN:  {sensitive_data._addresses['customer_ssn']}")
    print(f"    Model Weights: {sensitive_data._addresses['model_weights']}")
    time.sleep(0.5)
    
    print(f"{C.BLUE}[HYPERVISOR]{C.RESET} Reading physical memory pages...")
    time.sleep(0.5)
    
    print(f"\n{'─' * 78}")
    
    if is_tdx_protected:
        # TDX Protected - Show what hypervisor would see (encrypted)
        print(f"{C.GREEN}{C.BOLD}  MEMORY CONTENTS AS SEEN BY HYPERVISOR (TDX ENCRYPTED){C.RESET}")
        print(f"{'─' * 78}")
        print()
        
        print(f"  {C.CYAN}Address              Hex                                               ASCII{C.RESET}")
        print()
        
        # Show SIMULATED encrypted bytes (representing what hypervisor sees)
        # NOTE: This is simulated because TDX encryption is transparent to guest.
        # We cannot read actual encrypted RAM from within the VM.
        # TDX encrypts memory with AES-256-XTS before it reaches physical RAM.
        base_addr = int(sensitive_data._addresses['api_key'], 16)
        for i in range(6):
            addr = base_addr + (i * 16)
            encrypted_bytes = generate_encrypted_garbage(16)  # Simulated encrypted bytes
            hex_line = ' '.join(f'{b:02x}' for b in encrypted_bytes)
            ascii_repr = ''.join(chr(b) if 32 <= b < 127 else '.' for b in encrypted_bytes)
            print(f"  {C.DIM}{addr:016x}  {hex_line}  |{ascii_repr}|{C.RESET}")
            time.sleep(0.1)
        
        print()
        print(f"  {C.YELLOW}┌─────────────────────────────────────────────────────────────────────┐{C.RESET}")
        print(f"  {C.YELLOW}│ NOTE: Above bytes are SIMULATED - representing hypervisor's view  │{C.RESET}")
        print(f"  {C.YELLOW}│ We cannot read actual encrypted RAM from within the VM.           │{C.RESET}")
        print(f"  {C.YELLOW}│                                                                     │{C.RESET}")
        print(f"  {C.YELLOW}│ The actual data in VM memory is:                                   │{C.RESET}")
        print(f"  {C.YELLOW}│   API Key: {C.GREEN}{sensitive_data.api_key}{C.YELLOW}     │{C.RESET}")
        print(f"  {C.YELLOW}│                                                                     │{C.RESET}")
        print(f"  {C.YELLOW}│ TDX encrypts data at hardware level BEFORE it reaches RAM.        │{C.RESET}")
        print(f"  {C.YELLOW}└─────────────────────────────────────────────────────────────────────┘{C.RESET}")
        
        print()
        print(f"{'─' * 78}")
        print(f"""
  {C.GREEN}╔════════════════════════════════════════════════════════════════════════╗
  ║  🔒 TDX PROTECTION ACTIVE - ATTACK FAILED!                             ║
  ╠════════════════════════════════════════════════════════════════════════╣
  ║                                                                        ║
  ║  {C.BOLD}How TDX protects your data:{C.RESET}{C.GREEN}                                         ║
  ║                                                                        ║
  ║   VM Memory (plaintext)  →  CPU  →  Physical RAM (encrypted)          ║
  ║         ↑                     ↑              ↑                         ║
  ║    You see this          AES-256        Hypervisor sees this           ║
  ║                          encrypt                                       ║
  ║                                                                        ║
  ║  • Encryption keys are in CPU registers (inaccessible to hypervisor)  ║
  ║  • Each VM has unique per-boot encryption keys                        ║
  ║  • Hardware enforced - cannot be bypassed by software                 ║
  ║                                                                        ║
  ╚════════════════════════════════════════════════════════════════════════╝{C.RESET}
""")
        
    else:
        # Standard VM - Show ACTUAL plaintext data from REAL memory
        # This uses ctypes to read real bytes from actual memory addresses
        print(f"{C.RED}{C.BOLD}  MEMORY CONTENTS AS SEEN BY HYPERVISOR (PLAINTEXT){C.RESET}")
        print(f"{'─' * 78}")
        print()
        
        # Read ACTUAL memory content using ctypes (real memory, not simulated)
        print(f"  {C.CYAN}Reading REAL bytes from memory address {sensitive_data._addresses['api_key']}:{C.RESET}")
        print()
        
        # Use ctypes to read ACTUAL raw bytes from memory
        raw_memory = read_memory_bytes(sensitive_data.api_key, 96)
        if raw_memory:
            base_addr = int(sensitive_data._addresses['api_key'], 16)
            for line in format_memory_dump(raw_memory, base_addr):
                print(f"{C.RED}{line}{C.RESET}")
                time.sleep(0.1)
        else:
            # Fallback: show the string as bytes (still real data)
            api_key_bytes = sensitive_data.api_key.encode('utf-8')
            base_addr = int(sensitive_data._addresses['api_key'], 16)
            for line in format_memory_dump(api_key_bytes, base_addr):
                print(f"{C.RED}{line}{C.RESET}")
                time.sleep(0.1)
        
        print()
        print(f"  {C.RED}{C.BOLD}[EXTRACTED FROM RAW MEMORY BYTES]{C.RESET}")
        print()
        
        # Read and decode actual memory content for each field
        # This demonstrates that a hypervisor could scan memory and extract these values
        print(f"  {C.CYAN}Scanning memory for readable strings...{C.RESET}")
        time.sleep(0.3)
        
        # Read raw bytes and extract string data
        # Python strings store UTF-8 data after object header (offset ~48 bytes for str objects)
        def extract_string_from_memory(s):
            """Read string's actual bytes from memory and decode them"""
            raw = read_memory_bytes(s, 128)
            if raw:
                # Find printable ASCII/UTF-8 sequences in the memory
                # This simulates what a hypervisor memory scanner would do
                result = []
                for b in raw:
                    if 32 <= b < 127:  # Printable ASCII
                        result.append(chr(b))
                    elif result and b == 0:  # Null terminator
                        break
                extracted = ''.join(result)
                # Return the longest readable sequence
                return extracted if len(extracted) > 3 else s.encode('utf-8').hex()
            return None
        
        # Extract API key from memory
        api_key_raw = read_memory_bytes(sensitive_data.api_key, 96)
        print(f"\n  {C.RED}[MEMORY @ {sensitive_data._addresses['api_key']}]{C.RESET}")
        if api_key_raw:
            hex_preview = ' '.join(f'{b:02x}' for b in api_key_raw[:48])
            print(f"    Raw bytes: {C.DIM}{hex_preview}...{C.RESET}")
            extracted_key = extract_string_from_memory(sensitive_data.api_key)
            print(f"    Extracted: {C.WHITE}{extracted_key if extracted_key else 'N/A'}{C.RESET}")
        time.sleep(0.1)
        
        # Extract model name from memory
        model_addr = hex(id(sensitive_data.model_name))
        model_raw = read_memory_bytes(sensitive_data.model_name, 64)
        print(f"\n  {C.RED}[MEMORY @ {model_addr}]{C.RESET}")
        if model_raw:
            hex_preview = ' '.join(f'{b:02x}' for b in model_raw[:32])
            print(f"    Raw bytes: {C.DIM}{hex_preview}...{C.RESET}")
            extracted_model = extract_string_from_memory(sensitive_data.model_name)
            print(f"    Extracted: {C.WHITE}{extracted_model if extracted_model else 'N/A'}{C.RESET}")
        
        # Extract company from memory
        company_addr = hex(id(sensitive_data.company))
        company_raw = read_memory_bytes(sensitive_data.company, 64)
        print(f"\n  {C.RED}[MEMORY @ {company_addr}]{C.RESET}")
        if company_raw:
            hex_preview = ' '.join(f'{b:02x}' for b in company_raw[:32])
            print(f"    Raw bytes: {C.DIM}{hex_preview}...{C.RESET}")
            extracted_company = extract_string_from_memory(sensitive_data.company)
            print(f"    Extracted: {C.WHITE}{extracted_company if extracted_company else 'N/A'}{C.RESET}")
        
        # Read model weights from actual memory (list object)
        print(f"\n  {C.RED}[MEMORY @ {sensitive_data._addresses['model_weights']}]{C.RESET}")
        weights_raw = read_memory_bytes(sensitive_data.model_weights, 128)
        if weights_raw:
            hex_preview = ' '.join(f'{b:02x}' for b in weights_raw[:64])
            print(f"    Raw bytes: {C.DIM}{hex_preview}...{C.RESET}")
            # For floats, read each element's memory and decode using struct
            print(f"    Extracted float values from memory:")
            for i, weight_obj in enumerate(sensitive_data.model_weights):
                float_raw = read_memory_bytes(weight_obj, 32)
                if float_raw:
                    float_hex = ' '.join(f'{b:02x}' for b in float_raw[16:24])
                    # Decode float from IEEE 754 bytes (offset 16 in PyFloat object)
                    try:
                        decoded_float = struct.unpack('d', float_raw[16:24])[0]
                        print(f"      [{i}]: {C.DIM}{float_hex}{C.RESET} → {decoded_float:.4f}")
                    except:
                        print(f"      [{i}]: {C.DIM}{float_hex}{C.RESET}")
                time.sleep(0.02)
        time.sleep(0.1)
        
        # Read PII from memory
        print(f"\n  {C.RED}[MEMORY @ {sensitive_data._addresses['customer_ssn']}]{C.RESET}")
        ssn_raw = read_memory_bytes(sensitive_data.customer_ssn, 48)
        if ssn_raw:
            hex_preview = ' '.join(f'{b:02x}' for b in ssn_raw[:32])
            print(f"    Raw bytes: {C.DIM}{hex_preview}...{C.RESET}")
            extracted_ssn = extract_string_from_memory(sensitive_data.customer_ssn)
            print(f"    Extracted SSN: {C.WHITE}{extracted_ssn if extracted_ssn else 'N/A'}{C.RESET}")
        
        email_addr = hex(id(sensitive_data.customer_email))
        email_raw = read_memory_bytes(sensitive_data.customer_email, 64)
        print(f"\n  {C.RED}[MEMORY @ {email_addr}]{C.RESET}")
        if email_raw:
            hex_preview = ' '.join(f'{b:02x}' for b in email_raw[:32])
            print(f"    Raw bytes: {C.DIM}{hex_preview}...{C.RESET}")
            extracted_email = extract_string_from_memory(sensitive_data.customer_email)
            print(f"    Extracted Email: {C.WHITE}{extracted_email if extracted_email else 'N/A'}{C.RESET}")
        
        # Read inference prompt from memory
        prompt_addr = hex(id(sensitive_data.inference_prompt))
        prompt_raw = read_memory_bytes(sensitive_data.inference_prompt, 64)
        print(f"\n  {C.RED}[MEMORY @ {prompt_addr}]{C.RESET}")
        if prompt_raw:
            hex_preview = ' '.join(f'{b:02x}' for b in prompt_raw[:32])
            print(f"    Raw bytes: {C.DIM}{hex_preview}...{C.RESET}")
            extracted_prompt = extract_string_from_memory(sensitive_data.inference_prompt)
            print(f"    Extracted: {C.WHITE}{extracted_prompt if extracted_prompt else 'N/A'}{C.RESET}")
        time.sleep(0.1)
        
        print()
        print(f"{'─' * 78}")
        print(f"""
  {C.RED}╔════════════════════════════════════════════════════════════════════════╗
  ║  ⚠️  NO PROTECTION - ATTACK SUCCESSFUL!                                ║
  ╠════════════════════════════════════════════════════════════════════════╣
  ║                                                                        ║
  ║  {C.BOLD}Why this attack works:{C.RESET}{C.RED}                                               ║
  ║                                                                        ║
  ║   VM Memory (plaintext)  →  CPU  →  Physical RAM (PLAINTEXT!)         ║
  ║         ↑                              ↑                               ║
  ║    You see this                   Hypervisor sees SAME data            ║
  ║                                                                        ║
  ║  • No encryption between CPU and RAM                                  ║
  ║  • Hypervisor can read any VM's physical memory                       ║
  ║  • All secrets, keys, and data are exposed                            ║
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
  • Run: python3 3_tdx_attestation_demo.py  (to see remote attestation)
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

def run_demo_mode():
    """Demo mode: Show BOTH TDX and Standard VM scenarios for comparison"""
    clear_screen()
    print(f"""
{C.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║   {C.BOLD}TDX vs STANDARD VM - SIDE BY SIDE DEMONSTRATION{C.RESET}{C.CYAN}                           ║
║   {C.WHITE}Shows what hypervisor sees in BOTH environments{C.CYAN}                              ║
║                                                                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  {C.YELLOW}IMPORTANT: TDX encryption is transparent to the guest OS.{C.CYAN}                   ║
║  {C.YELLOW}• Standard VM scenario: Shows REAL memory reads via ctypes{C.CYAN}                  ║
║  {C.YELLOW}• TDX VM scenario: Shows SIMULATED encrypted bytes{C.CYAN}                          ║
║  {C.YELLOW}  (representing what hypervisor sees - cannot read real encrypted RAM){C.CYAN}      ║
╚══════════════════════════════════════════════════════════════════════════════╝{C.RESET}
""")
    
    # Allocate REAL sensitive data in memory
    print(f"  {C.CYAN}Allocating sensitive data in memory...{C.RESET}")
    sensitive_data = SensitiveData()
    print(f"  {C.GREEN}✓{C.RESET} Data allocated at real memory addresses")
    print(f"    API Key at:      {sensitive_data._addresses['api_key']}")
    print(f"    Model Weights at: {sensitive_data._addresses['model_weights']}")
    time.sleep(1)
    
    # Show attack scenario
    show_attack_scenario()
    
    # PART 1: Show Standard VM (Vulnerable) - using same data
    print(f"\n{C.YELLOW}{'═' * 78}{C.RESET}")
    print(f"{C.YELLOW}  SCENARIO 1: STANDARD VM (No TDX Protection){C.RESET}")
    print(f"{C.YELLOW}{'═' * 78}{C.RESET}")
    time.sleep(0.5)
    simulate_attack(is_tdx_protected=False, sensitive_data=sensitive_data)
    
    input(f"\n  {C.CYAN}Press Enter to see what happens with TDX protection...{C.RESET}")
    
    # PART 2: Show TDX VM (Protected) - same data, but encrypted
    print(f"\n{C.GREEN}{'═' * 78}{C.RESET}")
    print(f"{C.GREEN}  SCENARIO 2: TDX VM (Hardware Memory Encryption){C.RESET}")
    print(f"{C.GREEN}{'═' * 78}{C.RESET}")
    time.sleep(0.5)
    simulate_attack(is_tdx_protected=True, sensitive_data=sensitive_data)
    
    time.sleep(1)
    
    # Show comparison
    show_side_by_side_comparison()
    
    # Final summary
    print(f"\n{C.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}")
    print(f"  {C.BOLD}CONCLUSION:{C.RESET}")
    print(f"  • {C.RED}Standard VM{C.RESET}: Hypervisor can read ALL data in plaintext")
    print(f"  • {C.GREEN}TDX VM{C.RESET}: Hypervisor sees only encrypted garbage")
    print(f"  • {C.CYAN}TDX provides hardware-enforced protection that cannot be bypassed{C.RESET}")
    print(f"{C.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{C.RESET}")
    print()
    
    return 0


def main():
    """Main demo function"""
    # Check for --demo flag to show both scenarios
    demo_mode = '--demo' in sys.argv or '-d' in sys.argv
    
    if demo_mode:
        return run_demo_mode()
    
    # Normal mode: detect actual TDX status
    show_intro()
    
    # Detect TDX status
    print(f"  {C.CYAN}Detecting VM security status...{C.RESET}")
    time.sleep(1)
    tdx_status = detect_tdx_status()
    
    # Show current VM status
    is_tdx = show_vm_status(tdx_status)
    
    # Allocate REAL sensitive data in memory
    print(f"\n  {C.CYAN}Allocating sensitive data in memory...{C.RESET}")
    sensitive_data = SensitiveData()
    print(f"  {C.GREEN}✓{C.RESET} Data allocated at real memory addresses")
    print(f"    API Key at:      {sensitive_data._addresses['api_key']}")
    print(f"    Model Weights at: {sensitive_data._addresses['model_weights']}")
    time.sleep(1)
    
    # Show attack scenario
    show_attack_scenario()
    
    # Simulate the attack with real memory data
    simulate_attack(is_tdx, sensitive_data=sensitive_data)
    
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


def print_usage():
    print(f"""
{C.BOLD}Usage:{C.RESET} python3 2_memory_comparison_demo.py [OPTIONS]

{C.BOLD}Options:{C.RESET}
  (no args)     Detect actual TDX status and show appropriate output
  --demo, -d    Demo mode: Show BOTH scenarios (TDX + Standard) for comparison
  --help, -h    Show this help message

{C.BOLD}Examples:{C.RESET}
  python3 2_memory_comparison_demo.py          # Auto-detect mode
  python3 2_memory_comparison_demo.py --demo   # Show both scenarios
""")


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print_usage()
        sys.exit(0)
    
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Demo interrupted.{C.RESET}")
        sys.exit(130)
