#!/usr/bin/env python3
"""
Step 2: Sequential Security Scanning (Baseline)

This demonstrates traditional sequential scanning of ML models.
This is the SLOW approach that blocks inference.

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
"""

import os

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import sys
import json
import time
import hashlib
import re
import pickle
from pathlib import Path
from datetime import datetime

TEST_MODELS_DIR = Path("test_models")
RESULTS_DIR = Path("scan_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Check for simulation mode from .env (consistent with lab-07)
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "true").lower() == "true"

# Also check hardware status from amx_status.json for additional info
AMX_AVAILABLE = False
try:
    with open(".amx_status.json", "r") as f:
        amx_status = json.load(f)
        AMX_AVAILABLE = amx_status.get("amx_available", False)
except:
    pass

# In hardware mode, AMX must also be available
HARDWARE_MODE = (not SIMULATION_MODE) and AMX_AVAILABLE

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         Sequential Model Security Scanning (Baseline)                 ║
║              ⚠️  SLOW - Blocks inference pipeline ⚠️                  ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

def print_baseline_mode_indicator():
    """Print indicator that this is baseline (non-AMX) scanning."""
    print("\n" + "="*70)
    print("🔴 " + " BASELINE - Sequential (Non-Parallel) Scanning ".center(66, "=") + " 🔴")
    print("="*70)
    print("│ This is the SLOW baseline approach for comparison                │")
    print("│ No parallelization or AMX optimization                          │")
    if HARDWARE_MODE:
        print("│ 🟢 Note: Intel AMX IS available (HARDWARE MODE)                  │")
    elif AMX_AVAILABLE:
        print("│ 🔶 Note: Intel AMX available but SIMULATION_MODE=true            │")
    else:
        print("│ 🔶 Note: Intel AMX is NOT available (SIMULATION MODE)            │")
    print("="*70 + "\n")

# =============================================================================
# SECURITY SCANNING PATTERNS
# =============================================================================

DANGEROUS_PATTERNS = [
    # Code execution
    (r'\beval\s*\(', 'Dynamic code evaluation (eval)'),
    (r'\bexec\s*\(', 'Dynamic code execution (exec)'),
    (r'\bcompile\s*\(', 'Code compilation'),
    (r'__import__\s*\(', 'Dynamic import'),
    
    # System access
    (r'\bsubprocess\b', 'Subprocess module reference'),
    (r'\bos\.system\b', 'OS system call'),
    (r'\bos\.popen\b', 'OS popen call'),
    (r'commands\.', 'Commands module'),
    
    # Network
    (r'\bsocket\b', 'Socket operations'),
    (r'\burllib\b', 'URL library'),
    (r'\brequests\b', 'HTTP requests'),
    
    # Pickle exploits
    (r'__reduce__', 'Pickle reduce method (RCE vector)'),
    (r'__reduce_ex__', 'Pickle reduce_ex method'),
    
    # Shell
    (r'/bin/sh', 'Shell reference'),
    (r'/bin/bash', 'Bash reference'),
]

class SequentialModelScanner:
    """
    Traditional sequential model scanner.
    Scans models one at a time, blocking until complete.
    """
    
    def __init__(self):
        self.scan_results = []
        self.total_scan_time = 0
        
    def scan_file_patterns(self, filepath: Path) -> list:
        """Scan file for dangerous patterns."""
        findings = []
        
        try:
            # Read file content
            with open(filepath, 'rb') as f:
                content = f.read()
            
            # Try to decode as text for pattern matching
            try:
                text_content = content.decode('utf-8', errors='ignore')
            except:
                text_content = str(content)
            
            # Check each pattern
            for pattern, description in DANGEROUS_PATTERNS:
                matches = re.findall(pattern, text_content, re.IGNORECASE)
                if matches:
                    findings.append({
                        "pattern": pattern,
                        "description": description,
                        "matches": len(matches),
                        "severity": "HIGH" if "reduce" in pattern or "exec" in pattern else "MEDIUM"
                    })
        except Exception as e:
            findings.append({
                "error": str(e),
                "severity": "ERROR"
            })
        
        return findings
    
    def compute_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def analyze_pickle(self, filepath: Path) -> dict:
        """Analyze pickle file for suspicious content."""
        analysis = {
            "is_pickle": False,
            "suspicious": False,
            "details": []
        }
        
        if not filepath.suffix in ['.pkl', '.pickle']:
            return analysis
        
        analysis["is_pickle"] = True
        
        try:
            # Use fickling for pickle analysis if available
            try:
                from fickling.fickle import Fickling
                fickled = Fickling.load(str(filepath))
                if fickled.is_likely_safe:
                    analysis["details"].append("Pickle appears safe")
                else:
                    analysis["suspicious"] = True
                    analysis["details"].append("Pickle may contain unsafe operations")
            except ImportError:
                # Fallback: basic pickle inspection
                with open(filepath, 'rb') as f:
                    content = f.read()
                
                # Check for suspicious pickle opcodes
                if b'c__builtin__' in content or b'cos\n' in content:
                    analysis["suspicious"] = True
                    analysis["details"].append("Contains builtin/os references")
                
                if b'csubprocess' in content:
                    analysis["suspicious"] = True
                    analysis["details"].append("Contains subprocess reference")
                    
        except Exception as e:
            analysis["details"].append(f"Analysis error: {str(e)}")
        
        return analysis
    
    def scan_model(self, filepath: Path) -> dict:
        """Scan a single model file."""
        start_time = time.time()
        
        result = {
            "file": str(filepath),
            "filename": filepath.name,
            "size_bytes": filepath.stat().st_size,
            "scan_start": datetime.utcnow().isoformat(),
            "findings": [],
            "hash": None,
            "pickle_analysis": None,
            "status": "UNKNOWN"
        }
        
        # Step 1: Compute hash (simulates integrity check)
        result["hash"] = self.compute_hash(filepath)
        
        # Simulate slower scanning for demo purposes
        time.sleep(0.1)  # Simulated I/O delay
        
        # Step 2: Pattern matching
        result["findings"] = self.scan_file_patterns(filepath)
        
        # Simulate pattern matching time
        time.sleep(0.05 * len(DANGEROUS_PATTERNS))
        
        # Step 3: Pickle analysis
        result["pickle_analysis"] = self.analyze_pickle(filepath)
        
        # Determine status
        if result["findings"] or (result["pickle_analysis"] and result["pickle_analysis"].get("suspicious")):
            result["status"] = "SUSPICIOUS"
        else:
            result["status"] = "CLEAN"
        
        scan_time = time.time() - start_time
        result["scan_time_seconds"] = round(scan_time, 3)
        
        return result
    
    def scan_all(self, model_dir: Path) -> dict:
        """Scan all models in directory sequentially."""
        print("\n[*] Starting SEQUENTIAL scan...")
        print("    (Each model scanned one at a time, blocking)")
        print()
        
        overall_start = time.time()
        
        # Get all model files
        model_files = []
        for ext in ['*.h5', '*.keras', '*.pkl', '*.pickle', '*.pt', '*.pth']:
            model_files.extend(model_dir.glob(ext))
        
        model_files = sorted(model_files)
        total = len(model_files)
        
        print(f"    Found {total} models to scan")
        print("    " + "-"*50)
        
        for i, filepath in enumerate(model_files, 1):
            print(f"    [{i}/{total}] Scanning: {filepath.name}...", end=" ", flush=True)
            
            result = self.scan_model(filepath)
            self.scan_results.append(result)
            
            status_icon = "⚠️ " if result["status"] == "SUSPICIOUS" else "✓"
            print(f"{status_icon} {result['scan_time_seconds']:.2f}s")
            
            # Show findings if suspicious
            if result["status"] == "SUSPICIOUS":
                for finding in result["findings"][:2]:
                    print(f"        └─ {finding['description']}")
        
        self.total_scan_time = time.time() - overall_start
        
        return self.get_summary()
    
    def get_summary(self) -> dict:
        """Generate scan summary."""
        summary = {
            "scan_type": "SEQUENTIAL",
            "total_models": len(self.scan_results),
            "clean": sum(1 for r in self.scan_results if r["status"] == "CLEAN"),
            "suspicious": sum(1 for r in self.scan_results if r["status"] == "SUSPICIOUS"),
            "total_time_seconds": round(self.total_scan_time, 2),
            "avg_time_per_model": round(self.total_scan_time / len(self.scan_results), 3) if self.scan_results else 0,
            "results": self.scan_results
        }
        return summary

def show_blocking_demo():
    """Demonstrate how sequential scanning blocks inference."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    ⚠️  BLOCKING BEHAVIOR ⚠️                           ║
╚══════════════════════════════════════════════════════════════════════╝

During sequential scanning:

    ┌─────────────────────────────────────────────────────────────────┐
    │  Time ──────────────────────────────────────────────────────▶  │
    │                                                                 │
    │  SCAN:    [███ Model 1 ███][███ Model 2 ███][███ Model 3 ███]  │
    │                                                                 │
    │  INFERENCE: ╳ BLOCKED ╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳╳  │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

Problems:
  ❌ Inference pipeline blocked during entire scan
  ❌ Linear time complexity O(n) for n models
  ❌ CPU cores underutilized (single-threaded)
  ❌ Slow model onboarding in production
    """)

def main():
    print_banner()
    print_baseline_mode_indicator()
    
    if not TEST_MODELS_DIR.exists():
        print("[!] Test models not found. Run 1_generate_test_models.py first")
        sys.exit(1)
    
    show_blocking_demo()
    
    # Run sequential scan
    scanner = SequentialModelScanner()
    summary = scanner.scan_all(TEST_MODELS_DIR)
    
    # Save results
    results_path = RESULTS_DIR / "sequential_scan_results.json"
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SEQUENTIAL SCAN SUMMARY")
    print("="*60)
    print(f"  Total models scanned: {summary['total_models']}")
    print(f"  Clean models: {summary['clean']}")
    print(f"  Suspicious models: {summary['suspicious']}")
    print(f"  Total scan time: {summary['total_time_seconds']:.2f} seconds")
    print(f"  Avg time per model: {summary['avg_time_per_model']:.3f} seconds")
    print(f"  Results saved: {results_path}")
    print("="*60)
    
    print("""
    ⚠️  Note the scan time above. This is the BASELINE.
    
    With Intel AMX acceleration (next step), we can achieve:
    ├─ Parallel pattern matching
    ├─ Vectorized hash computation
    ├─ Non-blocking inference
    └─ 5-10x faster overall scanning
    
    Run: python 3_amx_parallel_scan.py to see the improvement
    """)

if __name__ == "__main__":
    main()
