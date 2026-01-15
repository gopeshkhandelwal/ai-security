"""
Model Security Scanner

A helper class to scan HuggingFace model cache for malicious code patterns
before loading with trust_remote_code=True.

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
from pathlib import Path


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
    ]
    
    TEXT_EXTENSIONS = ['.py', '.json', '.yaml', '.yml', '.txt', '.md', '.cfg', '.ini', '']
    BINARY_EXTENSIONS = ['.so', '.dll', '.dylib', '.bin', '.pkl', '.pickle']
    
    def __init__(self, model_cache_path: Path, verbose: bool = True):
        self.model_cache = Path(model_cache_path)
        self.verbose = verbose
        self.findings = []
        self.requires_custom_code = False
        self.auto_map_targets = []
        
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def check_custom_code_requirement(self) -> bool:
        """Check if model requires trust_remote_code=True."""
        self._log("[1/3] Checking if model requires custom code execution...")
        
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
        self._log(f"\n[2/3] Inspecting downloaded files...")
        
        all_files = list(self.model_cache.glob("*"))
        py_files = [f for f in all_files if f.suffix == '.py']
        binary_files = [f for f in all_files if f.suffix in self.BINARY_EXTENSIONS]
        other_files = [f for f in all_files if f not in py_files and f not in binary_files and f.is_file()]
        
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
        self._log(f"\n[3/3] Scanning downloaded code for red flags...")
        
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
    
    def scan(self) -> bool:
        """
        Run full security scan on model cache.
        
        Returns:
            True if model is safe to load, False if malicious code detected.
        """
        self.check_custom_code_requirement()
        files = self.inspect_downloaded_files()
        self.scan_for_malicious_patterns(files)
        return len(self.findings) == 0
    
    def print_assessment(self):
        """
        Print security assessment summary.
        
        """
        print("\n" + "=" * 60)
        print("  SECURITY ASSESSMENT")
        print("=" * 60)
        
        print("""
┌─────────────────────────────────────────────────────────────┐
│  🛡️  FIRST LINE OF DEFENSE                                  │
│                                                             │
│  AVOID models that require trust_remote_code=True           │
│                                                             │
│  • Use only models from verified publishers (Google, Meta)  │
│  • Prefer standard architectures (BERT, T5, GPT-2, Llama)   │
│  • If custom code is required, inspect ALL files first      │
└─────────────────────────────────────────────────────────────┘
""")
        
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
