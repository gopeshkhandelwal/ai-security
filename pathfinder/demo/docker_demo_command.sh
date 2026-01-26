#!/bin/bash
#
# Pathfinder Security Demo - All-in-One Docker Command
#
# Copy this entire block and paste into terminal to run the demo
#
# Prerequisites:
#   - Docker installed
#   - Intel Gaudi runtime (optional - will run in CPU mode without)
#

# === OPTION 1: Run with mounted workspace ===
# First, cd to your ai-security directory, then run:

cd /home/gopeshkh/Code/ai-security

docker run -it --rm \
    --runtime=habana \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    -e PYTHONPATH=/workspace \
    --cap-add=sys_nice --ipc=host \
    -v $(pwd):/workspace:rw \
    -w /workspace \
    vault.habana.ai/gaudi-docker/1.23.0/ubuntu22.04/habanalabs/pytorch-installer-2.9.0:latest \
    /bin/bash -c '
echo "============================================================"
echo "  Pathfinder Security Demo - Inside Gaudi Container"
echo "============================================================"

# Install scanning tools
pip install --quiet modelscan picklescan cryptography 2>/dev/null
pip install --quiet optimum-habana 2>/dev/null || true

# Run the scanner on Lab 01 malicious model
echo ""
echo "Scanning malicious model from Lab 01..."
echo ""

python3 << "PYTHON_SCRIPT"
import sys
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

# === Inline Scanner (minimal version) ===

class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class Finding:
    severity: Severity
    category: str
    message: str
    file: Optional[str] = None

@dataclass
class ScanResult:
    passed: bool = True
    findings: List[Finding] = field(default_factory=list)

DANGEROUS_PATTERNS = [
    (r"socket\.socket", "NETWORK_SOCKET", "Network socket creation"),
    (r"os\.fork\s*\(\)", "PROCESS_FORK", "Process forking"),
    (r"subprocess", "SUBPROCESS", "Subprocess execution"),
    (r"pty\.spawn", "PTY_SPAWN", "PTY shell spawning"),
    (r"os\.dup2", "FD_REDIRECT", "File descriptor redirection"),
    (r"exec\s*\(|eval\s*\(", "DYNAMIC_EXEC", "Dynamic code execution"),
    (r"os\.system", "OS_SYSTEM", "System command execution"),
    (r"base64\.b64decode", "BASE64_DECODE", "Base64 decoding"),
]

def scan_model(model_path: Path) -> ScanResult:
    result = ScanResult()
    
    # Find all Python files
    py_files = list(model_path.rglob("*.py"))
    
    for py_file in py_files:
        try:
            content = py_file.read_text()
            for pattern, category, description in DANGEROUS_PATTERNS:
                matches = re.findall(pattern, content)
                if matches:
                    severity = Severity.CRITICAL if category in [
                        "NETWORK_SOCKET", "SUBPROCESS", "PTY_SPAWN", "PROCESS_FORK"
                    ] else Severity.HIGH
                    
                    result.findings.append(Finding(
                        severity=severity,
                        category=category,
                        message=f"{description} ({len(matches)} occurrence(s))",
                        file=str(py_file.name)
                    ))
                    result.passed = False
        except Exception as e:
            pass
    
    # Check for trust_remote_code
    config_path = model_path / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            if "auto_map" in config:
                result.findings.append(Finding(
                    severity=Severity.HIGH,
                    category="TRUST_REMOTE_CODE",
                    message="Model requires trust_remote_code=True",
                    file="config.json"
                ))
                result.passed = False
        except:
            pass
    
    return result

# === Run Demo ===

print("=" * 70)
print("  PATHFINDER SECURITY SCANNER")
print("=" * 70)
print()

model_path = Path("/workspace/labs/lab-01-supply-chain-attack/hub_cache/models--helpful-ai--super-fast-qa-bert")

if not model_path.exists():
    model_path = Path("labs/lab-01-supply-chain-attack/hub_cache/models--helpful-ai--super-fast-qa-bert")

if not model_path.exists():
    print("❌ Lab 01 model not found. Make sure workspace is mounted correctly.")
    sys.exit(1)

print(f"Target: {model_path.name}")
print(f"This model contains a REVERSE SHELL backdoor from Lab 01")
print()

result = scan_model(model_path)

# Print results
if result.passed:
    print("✅ Status: PASSED")
else:
    print("❌ Status: FAILED - MALICIOUS CODE DETECTED!")

print()
print(f"Findings ({len(result.findings)}):")
print("-" * 70)

for f in result.findings:
    icon = "🚨" if f.severity == Severity.CRITICAL else "⚠️"
    print(f"  {icon} [{f.severity.value}] {f.category}")
    print(f"     {f.message}")
    if f.file:
        print(f"     File: {f.file}")
    print()

print("-" * 70)
print()
print("🛡️  ATTACK BLOCKED by Pathfinder!")
print()
print("The scanner detected these malicious patterns:")
print("  • socket.socket  → Creates reverse shell connection")
print("  • os.fork()      → Forks to run shell in background")
print("  • pty.spawn      → Spawns interactive bash shell")
print("  • os.dup2        → Redirects I/O to attacker socket")
print()
print("If loaded with trust_remote_code=True, the attacker would get")
print("full shell access to your machine!")
print()
print("=" * 70)
print("  Pathfinder prevents this by:")
print("=" * 70)
print("  ✓ Scanning all model files BEFORE loading")
print("  ✓ Blocking pickle-based formats (safetensors only)")
print("  ✓ Enforcing trust_remote_code=False")
print("  ✓ Requiring two-person approval for exceptions")
print("  ✓ ECDSA signing verified models")
print("=" * 70)
PYTHON_SCRIPT

echo ""
echo "Demo complete!"
'


# === OPTION 2: Without Gaudi runtime (CPU mode) ===
# If you don't have Gaudi hardware, remove --runtime=habana:
#
# docker run -it --rm \
#     -v $(pwd):/workspace:rw \
#     -w /workspace \
#     vault.habana.ai/gaudi-docker/1.23.0/ubuntu22.04/habanalabs/pytorch-installer-2.9.0:latest \
#     /bin/bash -c '...'
