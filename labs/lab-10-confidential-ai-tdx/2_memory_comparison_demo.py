#!/usr/bin/env python3
"""
Memory Encryption Demo: TDX vs Standard VM

Shows what a hypervisor sees when reading VM memory:
- Standard VM: Plaintext data exposed
- TDX VM: Encrypted garbage
"""

import os
import sys
import subprocess
import ctypes
import struct
import random

# Colors
RED = '\033[91m'
GREEN = '\033[92m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

def detect_tdx():
    """Check if TDX is enabled"""
    try:
        result = subprocess.run("sudo dmesg 2>/dev/null | grep -i 'tdx'", 
                                shell=True, capture_output=True, text=True, timeout=5)
        if "Guest detected" in result.stdout or "tdx_guest" in result.stdout.lower():
            return True
    except:
        pass
    return os.path.exists("/dev/tdx_guest")

def read_bytearray_memory(ba):
    """Read bytes directly from a bytearray's buffer"""
    buf_addr = (ctypes.c_char * len(ba)).from_buffer(ba)
    return bytes(buf_addr)

def main():
    print(f"\n{BOLD}Memory Encryption Demo{RESET}")
    print("=" * 50)
    
    is_tdx = detect_tdx()
    print(f"\nVM Status: {GREEN}TDX PROTECTED{RESET}" if is_tdx else f"\nVM Status: {RED}UNPROTECTED{RESET}")
    
    # Sensitive data in memory - use bytearray for direct memory access
    ssn = bytearray(b"123-45-6789")
    # Store weights as packed doubles in a buffer
    weights_list = [0.8234, -0.4521, 0.9182]
    weights_buf = bytearray(struct.pack('ddd', *weights_list))
    
    print(f"\n{BOLD}Hypervisor Memory Attack{RESET}")
    print("-" * 50)
    
    if is_tdx:
        # SIMULATION: TDX encrypts memory at the hardware level (CPU <-> RAM).
        # The random bytes below represent what a HYPERVISOR would see when
        # attempting to read physical RAM - encrypted date, not real data.
        print(f"\n{CYAN}1. Reading SSN (simulated hypervisor view):{RESET}")
        garbage = bytes([random.randint(0, 255) for _ in range(16)])
        print(f"   {GREEN}Encrypted: {garbage.hex()}{RESET}")
        
        print(f"\n{CYAN}2. Reading Model Weights (simulated hypervisor view):{RESET}")
        garbage = bytes([random.randint(0, 255) for _ in range(24)])
        print(f"   {GREEN}Encrypted: {garbage.hex()}{RESET}")
        
        print(f"\n{GREEN}ATTACK FAILED - Data protected by TDX{RESET}")
    else:
        # Standard: Hypervisor reads plaintext
        print(f"\n{CYAN}1. Reading SSN:{RESET}")
        raw = read_bytearray_memory(ssn)
        print(f"   {RED}Extracted: {raw.decode()}{RESET}")
        
        print(f"\n{CYAN}2. Reading Model Weights:{RESET}")
        raw = read_bytearray_memory(weights_buf)
        extracted_weights = struct.unpack('ddd', raw)
        for i, val in enumerate(extracted_weights):
            print(f"   {RED}[{i}]: {val:.4f}{RESET}")
        
        print(f"\n{RED}ATTACK SUCCESSFUL - All data exposed!{RESET}")
    
    print()

if __name__ == "__main__":
    main()
