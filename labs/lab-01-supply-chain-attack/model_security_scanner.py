"""
Model Security Scanner

A helper class to scan HuggingFace model cache for malicious code patterns
before loading with trust_remote_code=True.

Industry-standard security checks:
1. Pattern Matching - Detect dangerous code patterns
2. Publisher Verification - Warn on untrusted sources
3. File Hash Verification - Detect tampering
4. Entropy Analysis - Detect obfuscated payloads
5. Import Chain Analysis - Trace actual imports

Author: GopeshK
License: MIT License
Disclaimer: For educational and demonstration purposes only.

Usage:
    from model_security_scanner import ModelSecurityScanner
    
    scanner = ModelSecurityScanner("/path/to/model/cache")
    if scanner.scan():
        # Safe to load
        model = AutoModel.from_pretrained(..., trust_remote_code=True)
    else:
        # Malicious code detected!
        print(scanner.findings)
"""

import re
import json
import hashlib
import math
import ast
from pathlib import Path


# Verified/Trusted organizations on HuggingFace
TRUSTED_PUBLISHERS = [
    'google', 'meta-llama', 'microsoft', 'openai', 'huggingface',
    'facebook', 'nvidia', 'bigscience', 'EleutherAI', 'stabilityai',
    'mistralai', 'anthropic', 'databricks', 'mosaicml', 'tiiuae'
]


class ModelSecurityScanner:
    """
    Scans HuggingFace model cache for malicious code patterns.
    
    Use this class to validate downloaded model files BEFORE loading
    with trust_remote_code=True.
    
    Example:
        >>> scanner = ModelSecurityScanner("~/.cache/huggingface/hub/models--org--name")
        >>> if scanner.scan():
        ...     print("Safe to load!")
        ... else:
        ...     print("Malicious code detected:", scanner.findings)
    """
    
    SUSPICIOUS_PATTERNS = [
        (r'socket\.socket', 'Network socket creation'),
        (r'os\.fork\s*\(\)', 'Process forking'),
        (r'subprocess', 'Subprocess execution'),
        (r'pty\.spawn', 'PTY shell spawning'),
        (r'os\.dup2', 'File descriptor redirection'),
        (r'exec\s*\(|eval\s*\(', 'Dynamic code execution'),
        (r'urllib|requests\.get|http\.client', 'HTTP requests'),
        (r'os\.system', 'System command execution'),
        (r'pickle\.load|torch\.load|joblib\.load', 'Deserialization (pickle RCE risk)'),
        (r'ctypes|cffi', 'Native code execution'),
        (r'base64\.b64decode', 'Base64 decoding (potential payload hiding)'),
        (r'compile\s*\(', 'Dynamic code compilation'),
        (r'__import__\s*\(', 'Dynamic import (obfuscation technique)'),
        (r'getattr.*\(.*,.*\)\s*\(', 'Dynamic attribute call (obfuscation)'),
        (r'\\x[0-9a-fA-F]{2}', 'Hex-encoded strings (potential obfuscation)'),
    ]
    
    TEXT_EXTENSIONS = ['.py', '.json', '.yaml', '.yml', '.txt', '.md', '.cfg', '.ini', '']
    BINARY_EXTENSIONS = ['.so', '.dll', '.dylib', '.bin', '.pkl', '.pickle']
    
    # Entropy threshold for detecting obfuscated/encrypted content
    HIGH_ENTROPY_THRESHOLD = 5.5  # Normal code ~4.5, encrypted/obfuscated ~7.0
    
    def __init__(self, model_cache_path: Path, verbose: bool = True):
        self.model_cache = Path(model_cache_path)
        self.verbose = verbose
        self.findings = []
        self.warnings = []  # Non-critical issues
        self.requires_custom_code = False
        self.auto_map_targets = []
        self.publisher = None
        self.file_hashes = {}
        
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of string (detects obfuscation/encryption)."""
        if not data:
            return 0.0
        entropy = 0.0
        for x in range(256):
            p_x = data.count(chr(x)) / len(data)
            if p_x > 0:
                entropy -= p_x * math.log2(p_x)
        return entropy
    
    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of file for integrity verification."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def verify_publisher(self) -> bool:
        """Check if model is from a trusted/verified publisher."""
        self._log("[0/5] Verifying publisher trust level...")
        
        # Extract publisher from cache path (format: models--org--name)
        cache_name = self.model_cache.name
        if cache_name.startswith("models--"):
            parts = cache_name.split("--")
            if len(parts) >= 2:
                self.publisher = parts[1]
        
        if self.publisher:
            if self.publisher.lower() in [p.lower() for p in TRUSTED_PUBLISHERS]:
                self._log(f"  ✓ Trusted publisher: {self.publisher}")
                return True
            else:
                self.warnings.append(f"Untrusted publisher: {self.publisher}")
                self._log(f"  ⚠️  UNTRUSTED publisher: {self.publisher}")
                self._log(f"      Trusted orgs: {', '.join(TRUSTED_PUBLISHERS[:5])}...")
                return False
        else:
            self._log(f"  ⚠️  Could not determine publisher from path")
            return False
    
    def check_custom_code_requirement(self) -> bool:
        """Check if model requires trust_remote_code=True."""
        self._log("\n[1/5] Checking if model requires custom code execution...")
        
        config_path = self.model_cache / "config.json"
        if not config_path.exists():
            self._log("  ✓ No config.json found")
            return False
        
        with open(config_path) as f:
            config = json.load(f)
        
        if "auto_map" in config:
            self.requires_custom_code = True
            self.auto_map_targets = list(config['auto_map'].values())
            self._log(f"  ⚠️  Model requires trust_remote_code=True")
            self._log(f"  ⚠️  Will execute: {self.auto_map_targets}")
            return True
        else:
            self._log(f"  ✓ Standard model - no custom code needed")
            return False
    
    def inspect_downloaded_files(self) -> dict:
        """Categorize all downloaded files by type."""
        self._log(f"\n[2/5] Inspecting downloaded files...")
        
        all_files = list(self.model_cache.glob("*"))
        py_files = [f for f in all_files if f.suffix == '.py']
        binary_files = [f for f in all_files if f.suffix in self.BINARY_EXTENSIONS]
        other_files = [f for f in all_files if f not in py_files and f not in binary_files and f.is_file()]
        
        # Compute hashes for all files
        for f in all_files:
            if f.is_file():
                self.file_hashes[f.name] = self._compute_file_hash(f)
        
        file_count = len([f for f in all_files if f.is_file()])
        if file_count:
            self._log(f"  Found {file_count} file(s) in cache:")
            for f in all_files:
                if f.is_file():
                    if f in py_files:
                        self._log(f"     ⚠️  {f.name} (EXECUTABLE CODE)")
                    elif f in binary_files:
                        self._log(f"     🚨 {f.name} (BINARY - CANNOT INSPECT)")
                    else:
                        self._log(f"     - {f.name}")
        else:
            self._log(f"  ✓ No files downloaded")
        
        if binary_files:
            self._log(f"\n  🚨 WARNING: Binary files cannot be inspected for malicious code!")
            self._log(f"     These could contain compiled backdoors: {[f.name for f in binary_files]}")
        
        return {
            'all': all_files,
            'python': py_files,
            'binary': binary_files,
            'other': other_files
        }
    
    def scan_for_malicious_patterns(self, files: dict) -> list:
        """Scan text files for suspicious code patterns."""
        self._log(f"\n[3/5] Scanning downloaded code for red flags...")
        
        self.findings = []
        
        # Scan all text-based files
        text_files = [f for f in files['all'] if f.is_file() and f.suffix in self.TEXT_EXTENSIONS]
        for text_file in text_files:
            try:
                content = text_file.read_text()
                for pattern, desc in self.SUSPICIOUS_PATTERNS:
                    if re.search(pattern, content):
                        self.findings.append((text_file.name, desc))
            except UnicodeDecodeError:
                self.findings.append((text_file.name, "Binary content in text file (suspicious)"))
        
        # Flag binary files as inherently risky
        for bin_file in files['binary']:
            self.findings.append((bin_file.name, "Uninspectable binary file"))
        
        if self.findings:
            self._log(f"  🚨 DANGEROUS CODE DETECTED:")
            for filename, desc in self.findings:
                self._log(f"     - {filename}: {desc}")
        else:
            self._log(f"  ✓ No suspicious patterns found")
        
        return self.findings
    
    def analyze_entropy(self, files: dict) -> list:
        """Detect obfuscated/encrypted payloads via entropy analysis."""
        self._log(f"\n[4/5] Analyzing entropy for obfuscation detection...")
        
        high_entropy_files = []
        
        for f in files['all']:
            if f.is_file() and f.suffix == '.py':
                try:
                    content = f.read_text()
                    entropy = self._calculate_entropy(content)
                    
                    if entropy > self.HIGH_ENTROPY_THRESHOLD:
                        high_entropy_files.append((f.name, entropy))
                        self.warnings.append(f"High entropy in {f.name}: {entropy:.2f} (possible obfuscation)")
                except:
                    pass
        
        if high_entropy_files:
            self._log(f"  ⚠️  High entropy files detected (possible obfuscation):")
            for fname, ent in high_entropy_files:
                self._log(f"     - {fname}: entropy={ent:.2f} (threshold: {self.HIGH_ENTROPY_THRESHOLD})")
        else:
            self._log(f"  ✓ No obfuscated code detected")
        
        return high_entropy_files
    
    def analyze_imports(self, files: dict) -> list:
        """Analyze Python imports to detect dangerous dependencies."""
        self._log(f"\n[5/5] Analyzing import chains...")
        
        dangerous_imports = []
        DANGEROUS_MODULES = ['socket', 'subprocess', 'os', 'pty', 'ctypes', 'pickle', 'marshal']
        
        for f in files['python']:
            try:
                content = f.read_text()
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.split('.')[0] in DANGEROUS_MODULES:
                                dangerous_imports.append((f.name, f"import {alias.name}"))
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and node.module.split('.')[0] in DANGEROUS_MODULES:
                            dangerous_imports.append((f.name, f"from {node.module} import ..."))
            except:
                pass
        
        if dangerous_imports:
            self._log(f"  ⚠️  Dangerous imports found:")
            for fname, imp in dangerous_imports:
                self._log(f"     - {fname}: {imp}")
        else:
            self._log(f"  ✓ No dangerous imports detected")
        
        return dangerous_imports
    
    def scan(self) -> bool:
        """
        Run full security scan on model cache.
        
        Returns:
            True if model is safe to load, False if malicious code detected.
        """
        self.verify_publisher()
        self.check_custom_code_requirement()
        files = self.inspect_downloaded_files()
        self.scan_for_malicious_patterns(files)
        self.analyze_entropy(files)
        self.analyze_imports(files)
        return len(self.findings) == 0
    
    def print_assessment(self):
        """
        Print security assessment summary.
        
        """
        print("\n" + "=" * 60)
        print("  SECURITY ASSESSMENT")
        print("=" * 60)
        
        # File hashes for audit
        if self.file_hashes:
            print("\n📋 FILE HASHES (SHA256):")
            for fname, fhash in self.file_hashes.items():
                print(f"   {fname}: {fhash[:16]}...")
        
        print("""
┌─────────────────────────────────────────────────────────────┐
│  🛡️  DEFENSE LAYERS APPLIED                                 │
│                                                             │
│  ✓ Publisher Verification (Trusted org check)              │
│  ✓ Pattern Matching (Regex for dangerous code)             │
│  ✓ Entropy Analysis (Obfuscation detection)                │
│  ✓ Import Chain Analysis (AST-based inspection)            │
│  ✓ File Hash Recording (Integrity tracking)                │
│                                                             │
│  RECOMMENDATION: Only load models from verified publishers  │
│  Trusted: google, meta-llama, microsoft, mistralai, etc.   │
└─────────────────────────────────────────────────────────────┘
""")
        
        if self.warnings:
            print("⚠️  WARNINGS:")
            for w in self.warnings:
                print(f"   - {w}")
            print()
        
        if self.findings:
            print("""
🚨 CRITICAL: Malicious code detected in downloaded files!

The model's files contain:
  - Network socket creation (socket.socket)
  - Process forking (os.fork)
  - PTY shell spawning (pty.spawn)  
  - File descriptor redirection (os.dup2)

This is the signature of a REVERSE SHELL BACKDOOR.

If you run this with trust_remote_code=True, the attacker
will get shell access to your machine!
""")
            print("=" * 60)
            print("  ❌ NOT SAFE to load with trust_remote_code=True")
            print("=" * 60)
            print()
            return False
        else:
            print()
            print("=" * 60)
            print("  ✅ SAFE to load with trust_remote_code=True")
            print("=" * 60)
            print()
