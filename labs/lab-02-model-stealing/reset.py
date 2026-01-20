#!/usr/bin/env python3
"""
Reset Demo3 - Clean up generated files

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
            Do not use for malicious purposes. The author is not responsible
            for any misuse of this code.
"""

import os
import shutil

def reset():
    print("🧹 Resetting Demo3...")
    
    # Remove models directory
    if os.path.exists('models'):
        shutil.rmtree('models')
        print("   ✅ Removed models/")
    
    print("✅ Demo3 reset complete!")

if __name__ == "__main__":
    reset()
