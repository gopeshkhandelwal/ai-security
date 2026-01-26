#!/usr/bin/env python3
"""
Pathfinder Security Scan Demo

Demonstrates the security scanner catching malicious code
from Lab 01's supply chain attack model.

This script:
1. Scans a clean model (should PASS)
2. Scans the malicious model from Lab 01 (should FAIL with findings)
3. Shows how the scanner would block the attack
4. Demonstrates secure loading patterns

Run inside Gaudi Docker container:
    python security_scan_demo.py

Author: AI Model Pathfinder Security Team
"""

import os
import sys
import json
from pathlib import Path

# Add workspace root to Python path so imports work inside Docker
# Script location: /workspace/pathfinder/demo/security_scan_demo.py
# Workspace root: /workspace
SCRIPT_DIR = Path(__file__).parent.resolve()
PATHFINDER_DIR = SCRIPT_DIR.parent
WORKSPACE_ROOT = PATHFINDER_DIR.parent

# Add both workspace root and pathfinder dir to path
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(PATHFINDER_DIR))

# Import scanner
try:
    from pathfinder.security.pathfinder_scanner import PathfinderScanner, Severity
except ImportError:
    try:
        from security.pathfinder_scanner import PathfinderScanner, Severity
    except ImportError:
        # Final fallback - direct import
        sys.path.insert(0, str(PATHFINDER_DIR / "security"))
        from pathfinder_scanner import PathfinderScanner, Severity


def print_header(title: str):
    """Print a section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_result(result):
    """Pretty print scan result"""
    status = "✓ PASSED" if result.passed else "✗ FAILED"
    status_color = "\033[92m" if result.passed else "\033[91m"
    reset = "\033[0m"
    
    print(f"Status: {status_color}{status}{reset}")
    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Scanners: {', '.join(result.scanners_run)}")
    
    if result.scanners_skipped:
        print(f"Skipped: {', '.join(result.scanners_skipped)}")
    
    print(f"\nFormat Check:")
    print(f"  SafeTensors only: {result.format_check.get('safetensors_only', 'N/A')}")
    print(f"  Blocked files: {result.format_check.get('blocked_files', 0)}")
    
    if result.findings:
        print(f"\nFindings ({len(result.findings)}):")
        for f in result.findings:
            if f.severity == Severity.CRITICAL:
                icon, color = "🚨", "\033[91m"
            elif f.severity == Severity.HIGH:
                icon, color = "⚠️ ", "\033[93m"
            elif f.severity == Severity.MEDIUM:
                icon, color = "⚡", "\033[94m"
            else:
                icon, color = "ℹ️ ", "\033[90m"
            
            reset = "\033[0m"
            print(f"  {icon} {color}[{f.severity.value}]{reset} {f.category}")
            print(f"      {f.message}")
            if f.file:
                print(f"      File: {f.file}")


def demo_scan_malicious_model():
    """Demo: Scan the malicious model from Lab 01"""
    print_header("DEMO 1: Scanning Malicious Model (Lab 01)")
    
    # Path to Lab 01 malicious model
    malicious_model_path = Path("/workspace/labs/lab-01-supply-chain-attack/hub_cache/models--helpful-ai--super-fast-qa-bert")
    
    if not malicious_model_path.exists():
        # Try relative path
        malicious_model_path = Path("labs/lab-01-supply-chain-attack/hub_cache/models--helpful-ai--super-fast-qa-bert")
    
    if not malicious_model_path.exists():
        print(f"❌ Malicious model not found at: {malicious_model_path}")
        print("   Make sure you're running from the workspace root")
        return None
    
    print(f"Scanning: {malicious_model_path}")
    print(f"This model contains a REVERSE SHELL backdoor!\n")
    
    # Run scan
    scanner = PathfinderScanner(strict_mode=True, allow_pickle=False)
    result = scanner.scan_model(str(malicious_model_path))
    
    print_result(result)
    
    # Show the attack that would be blocked
    if not result.passed:
        print("\n" + "-" * 70)
        print("🛡️  ATTACK BLOCKED!")
        print("-" * 70)
        print("""
The scanner detected malicious patterns in this model:

  • socket.socket     → Creates network connection for reverse shell
  • os.fork()         → Forks process to run shell in background  
  • pty.spawn         → Spawns interactive PTY shell
  • os.dup2           → Redirects stdin/stdout to socket

If loaded with trust_remote_code=True, this model would:
  1. Connect to attacker's server
  2. Spawn a bash shell
  3. Give attacker full access to your machine!

Pathfinder BLOCKS this by:
  ✓ Scanning all .py files for dangerous patterns
  ✓ Detecting trust_remote_code requirement
  ✓ Requiring two-person approval for custom code
  ✓ Never allowing unscanned models to load
""")
    
    return result


def demo_scan_safe_model():
    """Demo: Show what a clean scan looks like"""
    print_header("DEMO 2: Creating & Scanning a Safe Model")
    
    # Create a temporary safe model structure
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "safe-model"
        model_dir.mkdir()
        
        # Create minimal safe model files
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 2048,
            "num_hidden_layers": 22,
            "vocab_size": 32000
            # Note: NO auto_map = no trust_remote_code needed
        }
        
        (model_dir / "config.json").write_text(json.dumps(config, indent=2))
        
        # Create a fake safetensors file (just for format check)
        (model_dir / "model.safetensors").write_bytes(b"SAFE" + b"\x00" * 100)
        
        # Create tokenizer config
        tokenizer_config = {"model_max_length": 2048}
        (model_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
        
        print(f"Created safe model at: {model_dir}")
        print("Files: config.json, model.safetensors, tokenizer_config.json")
        print("No custom Python code, uses safetensors format\n")
        
        # Run scan
        scanner = PathfinderScanner(strict_mode=True, allow_pickle=False)
        result = scanner.scan_model(str(model_dir))
        
        print_result(result)
        
        if result.passed:
            print("\n✅ This model is safe to load!")
            print("   • No dangerous code patterns detected")
            print("   • Uses safetensors format (no pickle RCE risk)")
            print("   • No trust_remote_code required")
        
        return result


def demo_show_secure_loading():
    """Demo: Show secure vs insecure loading patterns"""
    print_header("DEMO 3: Secure Loading Patterns")
    
    print("""
┌─────────────────────────────────────────────────────────────────────────┐
│  ❌ INSECURE (Current Practice)                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  from transformers import AutoModel                                     │
│                                                                         │
│  # DANGEROUS: Executes arbitrary Python from internet!                  │
│  model = AutoModel.from_pretrained(                                     │
│      "helpful-ai/super-fast-qa-bert",                                   │
│      trust_remote_code=True  # 🚨 RCE VULNERABILITY                     │
│  )                                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  ✅ SECURE (Pathfinder Pattern)                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  from pathfinder.security import PathfinderScanner, SecureModelLoader   │
│                                                                         │
│  # Step 1: Scan before loading                                          │
│  scanner = PathfinderScanner()                                          │
│  result = scanner.scan_model("/path/to/model")                          │
│                                                                         │
│  if not result.passed:                                                  │
│      raise SecurityError(f"Scan failed: {result.findings}")             │
│                                                                         │
│  # Step 2: Load with security constraints                               │
│  loader = SecureModelLoader(                                            │
│      verified_models_path="/verified-models",                           │
│      public_key_path="/keys/pathfinder.pub"                             │
│  )                                                                      │
│                                                                         │
│  load_result = loader.load("meta-llama/Llama-3.2-1B")                   │
│                                                                         │
│  # Internally enforces:                                                 │
│  #   - trust_remote_code=False (ALWAYS)                                 │
│  #   - local_files_only=True                                            │
│  #   - Signature verification                                           │
│  #   - MLBOM validation                                                 │
│                                                                         │
│  model = load_result.model                                              │
│  tokenizer = load_result.tokenizer                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
""")


def demo_gaudi_secure_inference():
    """Demo: Show secure inference pattern for Gaudi"""
    print_header("DEMO 4: Secure Gaudi Inference Pattern")
    
    print("""
┌─────────────────────────────────────────────────────────────────────────┐
│  Complete Secure Gaudi2 Inference Flow                                  │
└─────────────────────────────────────────────────────────────────────────┘

# 1. OUTSIDE CONTAINER: Enable model with security scan
$ pathfinder enable meta-llama/Llama-3.2-1B --gaudi-count=1

  [1/5] Preflight Validation
    ✓ Organization verified: meta-llama
    ✓ Format: safetensors only
    ✓ trust_remote_code: not required
  
  [2/5] Secure Download
    ✓ Downloaded to quarantine
    ✓ Checksum verified
  
  [3/5] Security Scan
    ✓ ModelScan: PASSED
    ✓ CodePatternScanner: PASSED
    ✓ PickleScan: PASSED
  
  [4/5] Sign & Verify  
    ✓ ECDSA signature generated
    ✓ MLBOM created
  
  [5/5] Model promoted to /verified-models/

# 2. INSIDE SECURE CONTAINER: Run inference
$ docker run --runtime=habana \\
    --network=pathfinder-isolated \\
    -v /verified-models:/models:ro \\
    pathfinder/gaudi-secure:1.23.0

# Inside container:
python << 'EOF'
import torch
import habana_frameworks.torch as ht
from pathfinder.security import SecureModelLoader

# Secure load (verifies signature + MLBOM)
loader = SecureModelLoader(
    verified_models_path="/models",
    public_key_path="/keys/pathfinder.pub",
    device="hpu"
)

result = loader.load("meta-llama/Llama-3.2-1B")
model = result.model
tokenizer = result.tokenizer

# Run inference on Gaudi
inputs = tokenizer("Hello, AI!", return_tensors="pt").to("hpu")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
EOF

# 3. Security guarantees:
#    ✓ Model verified via ECDSA signature
#    ✓ MLBOM confirms provenance
#    ✓ trust_remote_code=False enforced
#    ✓ Network egress blocked (except allowlist)
#    ✓ Read-only model mount
#    ✓ Secrets injected via mount (not env vars)
""")


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║              AI Model Pathfinder - Security Scanning Demo                 ║
║                                                                           ║
║  This demo shows how Pathfinder protects against:                         ║
║    • Supply chain attacks (malicious model code)                          ║
║    • Pickle deserialization RCE                                           ║
║    • trust_remote_code exploitation                                       ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")
    
    # Run demos
    demo_scan_malicious_model()
    demo_scan_safe_model()
    demo_show_secure_loading()
    demo_gaudi_secure_inference()
    
    print_header("Summary")
    print("""
Pathfinder Security Pipeline:

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  PREFLIGHT   │ ──▶ │    SCAN      │ ──▶ │    SIGN      │
  │  Allowlist   │     │  ModelScan   │     │   ECDSA      │
  │  Format      │     │  PickleScan  │     │   MLBOM      │
  │  GPG         │     │  AST Scan    │     │   Hash       │
  └──────────────┘     └──────────────┘     └──────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │   VERIFY     │
                       │  at runtime  │
                       │  on Gaudi    │
                       └──────────────┘

Key Security Controls:
  • trust_remote_code=False (ALWAYS enforced)
  • safetensors-only format (no pickle RCE)
  • ECDSA signatures on all artifacts
  • MLBOM for provenance tracking
  • Container isolation with egress rules
  • Two-person approval for exceptions

Run this demo yourself:
  cd /workspace
  ./pathfinder/demo/run_security_demo.sh
""")


if __name__ == "__main__":
    main()
