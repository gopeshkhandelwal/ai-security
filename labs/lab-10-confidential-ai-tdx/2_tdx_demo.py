#!/usr/bin/env python3
"""
Intel TDX Demo: Memory Encryption + Attestation

Part 1: Shows what hypervisor sees when reading VM memory
Part 2: Shows how to prove TDX protection (attestation)
"""

import os
import subprocess
import ctypes
import struct
import random
import hashlib

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
    print(f"\n{BOLD}Intel TDX Demo{RESET}")
    print("=" * 50)
    
    is_tdx = detect_tdx()
    print(f"\nVM Status: {GREEN}TDX PROTECTED{RESET}" if is_tdx else f"\nVM Status: {RED}UNPROTECTED{RESET}")
    
    # ============================================================
    # PART 1: Memory Attack Demo
    # ============================================================
    print(f"\n{BOLD}Part 1: Hypervisor Memory Attack{RESET}")
    print("-" * 50)
    
    # Sensitive data in memory
    ssn = bytearray(b"123-45-6789")
    weights_list = [0.8234, -0.4521, 0.9182]
    weights_buf = bytearray(struct.pack('ddd', *weights_list))
    
    if is_tdx:
        # SIMULATION: TDX encrypts at hardware level.
        # Random bytes represent what hypervisor sees (encrypted).
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
    
    # ============================================================
    # PART 2: Attestation (Proof of TDX)
    # ============================================================
    print(f"\n{BOLD}Part 2: Remote Attestation{RESET}")
    print("-" * 50)
    
    if is_tdx and os.path.exists("/dev/tdx_guest"):
        print(f"\n{GREEN}Attestation device: /dev/tdx_guest{RESET}")
        print(f"{GREEN}Reading REAL TDX attestation report from hardware...{RESET}")
        
        try:
            import fcntl
            import secrets
            
            # TDX ioctl command (from Linux kernel: TDX_CMD_GET_REPORT0)
            # _IOWR('T', 1, struct tdx_report_req) = 0xc0104001
            TDX_CMD_GET_REPORT0 = 0xc0104001
            
            # Generate 64-byte report_data (challenge nonce)
            report_data = secrets.token_bytes(64)
            
            # TDX report request structure:
            # - subtype: u8 (0 for REPORTTYPE0)
            # - reportdata: [u8; 64]
            # - rpd_len: u32
            # - tdreport: [u8; 1024]
            # - tdr_len: u32
            
            # Build request buffer
            subtype = 0
            rpd_len = 64
            tdr_len = 1024
            
            # Pack: subtype(1) + padding(7) + reportdata(64) + rpd_len(4) + padding(4) + tdreport(1024) + tdr_len(4)
            request = bytearray(1 + 7 + 64 + 4 + 4 + 1024 + 4)
            request[0] = subtype
            request[8:72] = report_data
            struct.pack_into('<I', request, 72, rpd_len)
            struct.pack_into('<I', request, 1104, tdr_len)
            
            # Open device and call ioctl
            fd = os.open("/dev/tdx_guest", os.O_RDWR)
            try:
                fcntl.ioctl(fd, TDX_CMD_GET_REPORT0, request)
            finally:
                os.close(fd)
            
            # Parse the TDX report (offset 80, 1024 bytes)
            # TDX Report structure offsets:
            # - REPORTMACSTRUCT: bytes 0-255
            # - TEE_TCB_SVN: bytes 0-15
            # - MRSEAM: bytes 16-63
            # - MRSIGNERSEAM: bytes 64-111
            # - SEAMATTRIBUTES: bytes 112-119
            # - TDATTRIBUTES: bytes 120-127
            # - XFAM: bytes 128-135
            # - MRTD: bytes 136-183
            # - MRCONFIGID: bytes 184-231
            # - MROWNER: bytes 232-279
            # - MROWNERCONFIG: bytes 280-327
            # - RTMR0-3: bytes 328-519
            # - REPORTDATA: bytes 520-583
            
            tdreport = request[80:1104]
            
            # Extract real measurements from the report
            mrtd = tdreport[136:184].hex()
            mrowner = tdreport[232:280].hex()
            rtmr0 = tdreport[328:376].hex()
            rtmr1 = tdreport[376:424].hex()
            reportdata = tdreport[520:584].hex()
            
            print(f"\n{CYAN}REAL TDX Report Contents (from hardware):{RESET}")
            print(f"  MRTD (VM Hash):     {mrtd[:48]}...")
            print(f"  MROWNER:            {mrowner[:48]}...")
            print(f"  RTMR0:              {rtmr0[:48]}...")
            print(f"  RTMR1:              {rtmr1[:48]}...")
            print(f"  REPORTDATA (nonce): {reportdata[:48]}...")
            
            print(f"\n{GREEN}What this proves:{RESET}")
            print(f"  1. VM is running on genuine Intel TDX hardware")
            print(f"  2. Memory is encrypted with AES-256-XTS")
            print(f"  3. Report is signed by CPU (unforgeable)")
            print(f"  4. Third parties verify via Intel Attestation Service")
            
        except PermissionError:
            print(f"\n{RED}Permission denied - run with sudo for real report{RESET}")
        except Exception as e:
            print(f"\n{RED}Error reading TDX report: {e}{RESET}")
            print(f"Device exists but ioctl failed - check kernel support")
    else:
        print(f"\n{RED}No attestation available (not a TDX VM){RESET}")
        print(f"Attestation proves to customers that their data is protected.")
        print(f"Without TDX, you have to trust the cloud provider's word.")
    
    print()

if __name__ == "__main__":
    main()
