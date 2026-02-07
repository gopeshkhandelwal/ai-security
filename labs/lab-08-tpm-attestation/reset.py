#!/usr/bin/env python3
"""
Reset Lab 08: TPM Attestation for AI Models

Cleans up all generated files from the lab.
Note: TPM PCR values cannot be reset without a system reboot.

Author: GopeshK
License: MIT License
"""

import os
import glob
from pathlib import Path


def main():
    print("Resetting Lab 08: TPM Attestation for AI Models")
    print("=" * 50)
    
    # Files to remove
    files_to_remove = [
        # TPM status
        ".tpm_status.json",
        
        # Model files
        "test_model.h5",
        "tampered_model.h5",
        
        # Measurement files
        "model_measurement.json",
        
        # Quote files
        "quote.msg",
        "quote.sig", 
        "quote.pcrs",
        
        # Attestation Identity Key files
        "aik.pub",
        "aik.priv",
        "aik.ctx",
        "primary.ctx",
        
        # Attestation results
        "attestation_package.json",
        "attestation_result.json",
        "attestation_token.jwt",
    ]
    
    removed = 0
    for filename in files_to_remove:
        filepath = Path(filename)
        if filepath.exists():
            try:
                filepath.unlink()
                print(f"  Removed: {filename}")
                removed += 1
            except Exception as e:
                print(f"  Failed to remove {filename}: {e}")
    
    # Remove any .h5 files created during demo
    for h5_file in glob.glob("*.h5"):
        try:
            os.remove(h5_file)
            print(f"  Removed: {h5_file}")
            removed += 1
        except Exception as e:
            print(f"  Failed to remove {h5_file}: {e}")
    
    print(f"\nRemoved {removed} files.")
    
    print("""
Note: TPM PCR values cannot be reset programmatically.
      PCRs are reset on system reboot.
      This is a security feature - you cannot undo measurements.
    """)
    
    print("Lab reset complete!")


if __name__ == "__main__":
    main()
