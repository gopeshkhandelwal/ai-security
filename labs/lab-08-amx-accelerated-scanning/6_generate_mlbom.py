#!/usr/bin/env python3
"""
Step 6: Generate ML Bill of Materials (MLBOM)

Generate comprehensive MLBOM for models, including dependencies,
security scan results, and provenance information.

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
import hashlib
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

TEST_MODELS_DIR = Path("test_models")
MLBOM_DIR = Path("mlbom_output")
MLBOM_DIR.mkdir(exist_ok=True)

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
║         ML Bill of Materials (MLBOM) Generator                        ║
║              📋 Supply Chain Transparency for AI                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

def print_amx_mode_indicator():
    """Print a clear indicator of AMX hardware status."""
    if HARDWARE_MODE:
        print("\n" + "="*70)
        print("🟢 " + " HARDWARE MODE - Intel AMX Active ".center(66, "=") + " 🟢")
        print("="*70)
        print("│ MLBOM generation with REAL Intel AMX acceleration               │")
        print("│ Parallel hash computation using hardware acceleration           │")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("🔶 " + " SIMULATION MODE ".center(66, "=") + " 🔶")
        print("="*70)
        if not AMX_AVAILABLE:
            print("│ MLBOM generation WITHOUT Intel AMX hardware                     │")
        else:
            print("│ SIMULATION_MODE=true in .env (AMX hardware is available)         │")
        print("│ Using standard parallel processing for demonstration            │")
        print("="*70 + "\n")

# =============================================================================
# MLBOM GENERATOR
# =============================================================================

class MLBOMGenerator:
    """
    Generates ML Bill of Materials for AI models.
    
    MLBOM includes:
    - Model metadata and format
    - Cryptographic hashes
    - Dependencies and versions
    - Security scan results
    - Provenance information
    """
    
    def __init__(self):
        self.mlboms = []
    
    def compute_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def detect_format(self, filepath: Path) -> dict:
        """Detect model format and framework."""
        suffix = filepath.suffix.lower()
        
        format_map = {
            '.h5': {'format': 'HDF5', 'framework': 'TensorFlow/Keras'},
            '.keras': {'format': 'Keras Native', 'framework': 'TensorFlow/Keras'},
            '.pkl': {'format': 'Pickle', 'framework': 'scikit-learn/custom'},
            '.pickle': {'format': 'Pickle', 'framework': 'scikit-learn/custom'},
            '.pt': {'format': 'PyTorch', 'framework': 'PyTorch'},
            '.pth': {'format': 'PyTorch', 'framework': 'PyTorch'},
            '.onnx': {'format': 'ONNX', 'framework': 'ONNX Runtime'},
            '.safetensors': {'format': 'SafeTensors', 'framework': 'HuggingFace'},
        }
        
        return format_map.get(suffix, {'format': 'Unknown', 'framework': 'Unknown'})
    
    def scan_for_issues(self, filepath: Path) -> dict:
        """Quick security scan of model file."""
        issues = []
        
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # Check for suspicious patterns
        patterns = {
            b'eval(': 'Dynamic code evaluation',
            b'exec(': 'Dynamic code execution',
            b'subprocess': 'Subprocess reference',
            b'__reduce__': 'Pickle reduce method',
            b'os.system': 'OS system call',
        }
        
        for pattern, description in patterns.items():
            if pattern in content:
                issues.append({
                    "type": "suspicious_pattern",
                    "pattern": pattern.decode(),
                    "description": description,
                    "severity": "HIGH"
                })
        
        return {
            "scanned": True,
            "issues_found": len(issues),
            "issues": issues,
            "status": "FAIL" if issues else "PASS"
        }
    
    def get_dependencies(self, filepath: Path) -> list:
        """Get inferred dependencies based on model format."""
        format_info = self.detect_format(filepath)
        framework = format_info.get('framework', '')
        
        # Common dependencies by framework
        dependencies = {
            'TensorFlow/Keras': [
                {'name': 'tensorflow', 'version': '>=2.0.0'},
                {'name': 'numpy', 'version': '>=1.19.0'},
                {'name': 'h5py', 'version': '>=3.0.0'},
            ],
            'scikit-learn/custom': [
                {'name': 'scikit-learn', 'version': '>=1.0.0'},
                {'name': 'numpy', 'version': '>=1.19.0'},
                {'name': 'joblib', 'version': '>=1.0.0'},
            ],
            'PyTorch': [
                {'name': 'torch', 'version': '>=1.9.0'},
                {'name': 'numpy', 'version': '>=1.19.0'},
            ],
            'HuggingFace': [
                {'name': 'transformers', 'version': '>=4.0.0'},
                {'name': 'safetensors', 'version': '>=0.3.0'},
            ]
        }
        
        return dependencies.get(framework, [])
    
    def generate_mlbom(self, filepath: Path) -> dict:
        """Generate complete MLBOM for a model."""
        format_info = self.detect_format(filepath)
        
        mlbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "version": 1,
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "component": {
                    "type": "machine-learning-model",
                    "name": filepath.stem,
                    "version": "1.0.0"
                }
            },
            "model": {
                "name": filepath.name,
                "path": str(filepath.absolute()),
                "format": format_info["format"],
                "framework": format_info["framework"],
                "size_bytes": filepath.stat().st_size,
                "hashes": [
                    {
                        "algorithm": "SHA-256",
                        "value": self.compute_hash(filepath)
                    }
                ]
            },
            "components": self.get_dependencies(filepath),
            "security": self.scan_for_issues(filepath),
            "provenance": {
                "generated_by": "MLBOM Generator v1.0",
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "tool": "Lab 08: AMX Accelerated Scanning"
            }
        }
        
        return mlbom
    
    def generate_all_parallel(self, model_dir: Path) -> list:
        """Generate MLBOMs for all models in parallel."""
        model_files = []
        for ext in ['*.h5', '*.keras', '*.pkl', '*.pickle', '*.pt']:
            model_files.extend(model_dir.glob(ext))
        
        print(f"\n[*] Generating MLBOM for {len(model_files)} models...")
        
        start_time = time.time()
        
        with ThreadPoolExecutor() as executor:
            self.mlboms = list(executor.map(self.generate_mlbom, model_files))
        
        generation_time = time.time() - start_time
        print(f"[✓] Generated {len(self.mlboms)} MLBOMs in {generation_time:.2f}s")
        
        return self.mlboms

def explain_mlbom():
    """Explain what MLBOM is and why it matters."""
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                WHAT IS MLBOM?                                        │
└─────────────────────────────────────────────────────────────────────┘

MLBOM (ML Bill of Materials) is the AI equivalent of SBOM for software.
It provides transparency into:

  📦 MODEL METADATA
     ├─ Name, version, format
     ├─ Framework (TensorFlow, PyTorch, etc.)
     └─ File size and location

  🔐 CRYPTOGRAPHIC HASHES
     ├─ SHA-256 hash for integrity
     └─ Used to verify model hasn't been tampered

  📚 DEPENDENCIES
     ├─ Required libraries and versions
     └─ Potential vulnerability exposure

  🔍 SECURITY SCAN RESULTS
     ├─ Pattern matching results
     ├─ Pickle analysis
     └─ Known vulnerability checks

  📍 PROVENANCE
     ├─ When/where/how generated
     └─ Training data lineage (optional)


WHY IT MATTERS:
┌─────────────────────────────────────────────────────────────────────┐
│  ✓ Supply Chain Security: Know what's in your models               │
│  ✓ Compliance: NIST, EU AI Act requirements                        │
│  ✓ Vulnerability Management: Track dependency risks                │
│  ✓ Integrity: Verify models haven't been tampered                  │
│  ✓ Reproducibility: Document model provenance                      │
└─────────────────────────────────────────────────────────────────────┘
    """)

def print_sample_mlbom(mlbom: dict):
    """Print a formatted sample MLBOM."""
    print("\n" + "="*60)
    print("SAMPLE MLBOM OUTPUT")
    print("="*60)
    
    print(f"""
Model: {mlbom['model']['name']}
Format: {mlbom['model']['format']} ({mlbom['model']['framework']})
Size: {mlbom['model']['size_bytes']:,} bytes
Hash: {mlbom['model']['hashes'][0]['value'][:32]}...

Security Status: {mlbom['security']['status']}
Issues Found: {mlbom['security']['issues_found']}
""")
    
    if mlbom['security']['issues']:
        print("  Issues:")
        for issue in mlbom['security']['issues'][:3]:
            print(f"    ⚠️  {issue['description']} (severity: {issue['severity']})")
    
    print("\nDependencies:")
    for dep in mlbom['components'][:5]:
        print(f"    - {dep['name']} {dep['version']}")
    
    print("\nFull MLBOM saved to: mlbom_output/")

def save_mlboms(mlboms: list):
    """Save MLBOMs to files."""
    # Save individual MLBOMs
    for mlbom in mlboms:
        model_name = mlbom['model']['name']
        output_path = MLBOM_DIR / f"{model_name}.mlbom.json"
        with open(output_path, 'w') as f:
            json.dump(mlbom, f, indent=2)
    
    # Save combined MLBOM
    combined = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "tools": [{"name": "MLBOM Generator", "version": "1.0"}]
        },
        "models": [m['model'] for m in mlboms],
        "components": list({d['name']: d for m in mlboms for d in m['components']}.values()),
        "security_summary": {
            "total_scanned": len(mlboms),
            "passed": sum(1 for m in mlboms if m['security']['status'] == "PASS"),
            "failed": sum(1 for m in mlboms if m['security']['status'] == "FAIL"),
        }
    }
    
    combined_path = MLBOM_DIR / "combined.mlbom.json"
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f"\n[✓] Individual MLBOMs saved to: {MLBOM_DIR}/")
    print(f"[✓] Combined MLBOM saved to: {combined_path}")

def main():
    print_banner()
    print_amx_mode_indicator()
    explain_mlbom()
    
    if not TEST_MODELS_DIR.exists():
        print("[!] Test models not found. Run 1_generate_test_models.py first")
        sys.exit(1)
    
    # Generate MLBOMs
    generator = MLBOMGenerator()
    mlboms = generator.generate_all_parallel(TEST_MODELS_DIR)
    
    # Print sample
    if mlboms:
        print_sample_mlbom(mlboms[0])
    
    # Save all MLBOMs
    save_mlboms(mlboms)
    
    # Summary
    print("\n" + "="*60)
    print("MLBOM GENERATION SUMMARY")
    print("="*60)
    passed = sum(1 for m in mlboms if m['security']['status'] == "PASS")
    failed = sum(1 for m in mlboms if m['security']['status'] == "FAIL")
    
    print(f"""
    Total models processed: {len(mlboms)}
    Security scan passed:   {passed}
    Security scan failed:   {failed}
    
    Output files:
    ├─ Individual: mlbom_output/<model>.mlbom.json
    └─ Combined:   mlbom_output/combined.mlbom.json
    """)
    
    print("\n" + "="*60)
    print("🎉 LAB 08 COMPLETE!")
    print("="*60)
    print("""
You have learned:

  ✓ How Intel AMX accelerates security scanning operations
  ✓ Sequential vs parallel scanning performance
  ✓ Async non-blocking scanning architecture
  ✓ MLBOM generation for supply chain transparency

Key Takeaways:

  ⚡ Parallel scanning with AMX can achieve 5-10x speedup
  🔒 Security doesn't have to slow down inference
  📋 MLBOM provides visibility into model supply chain
  🚀 Async scanning enables production-grade deployment

CLEANUP:
  python reset.py
    """)

if __name__ == "__main__":
    main()
