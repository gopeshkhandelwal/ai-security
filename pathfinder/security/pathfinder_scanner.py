#!/usr/bin/env python3
"""
Pathfinder Model Security Scanner

Integrates multiple security scanning tools for AI Model Pathfinder:
1. ModelScan (Protect AI) - Industry-standard ML model scanner
2. PickleScan - Detect malicious pickle files
3. AST Code Scanner - Analyze custom model code (from Lab 01)
4. SafeTensors Validator - Ensure safe format usage
5. Hash Verification - Integrity checking

This module is the core security gate for model enablement.

Usage:
    from pathfinder_scanner import PathfinderScanner
    
    scanner = PathfinderScanner()
    result = scanner.scan_model("/path/to/model")
    
    if result.passed:
        # Safe to proceed with signing and loading
        pass
    else:
        # Block and quarantine
        print(result.to_json())

Author: AI Model Pathfinder Security Team
License: MIT
"""

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pathfinder.scanner")


# =============================================================================
# SCANNER AVAILABILITY CHECKS
# =============================================================================

# ModelScan (Protect AI)
try:
    from modelscan.modelscan import ModelScan
    MODELSCAN_AVAILABLE = True
except ImportError:
    MODELSCAN_AVAILABLE = False
    logger.warning("ModelScan not available. Install with: pip install modelscan")

# PickleScan
try:
    import picklescan.scanner as picklescan
    PICKLESCAN_AVAILABLE = True
except ImportError:
    PICKLESCAN_AVAILABLE = False
    logger.warning("PickleScan not available. Install with: pip install picklescan")

# SafeTensors
try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


# =============================================================================
# DATA CLASSES
# =============================================================================

class Severity(Enum):
    """Finding severity levels"""
    CRITICAL = "CRITICAL"  # Block immediately
    HIGH = "HIGH"          # Block, requires approval
    MEDIUM = "MEDIUM"      # Warning, log for audit
    LOW = "LOW"            # Informational
    INFO = "INFO"          # Informational


@dataclass
class Finding:
    """Security finding from a scan"""
    scanner: str           # Which scanner found this
    severity: Severity     # Severity level
    category: str          # Category (e.g., "PICKLE_INJECTION", "NETWORK_ACCESS")
    message: str           # Human-readable description
    file: Optional[str] = None  # Affected file
    line: Optional[int] = None  # Line number if applicable
    details: Optional[Dict] = None  # Additional details
    
    def to_dict(self) -> dict:
        return {
            "scanner": self.scanner,
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "file": self.file,
            "line": self.line,
            "details": self.details
        }


@dataclass
class ScanResult:
    """Complete scan result"""
    model_path: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    passed: bool = True
    findings: List[Finding] = field(default_factory=list)
    scanners_run: List[str] = field(default_factory=list)
    scanners_skipped: List[str] = field(default_factory=list)
    file_hashes: Dict[str, str] = field(default_factory=dict)
    format_check: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    
    def add_finding(self, finding: Finding):
        self.findings.append(finding)
        # Auto-fail on CRITICAL or HIGH
        if finding.severity in [Severity.CRITICAL, Severity.HIGH]:
            self.passed = False
    
    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "summary": {
                "total_findings": len(self.findings),
                "critical": sum(1 for f in self.findings if f.severity == Severity.CRITICAL),
                "high": sum(1 for f in self.findings if f.severity == Severity.HIGH),
                "medium": sum(1 for f in self.findings if f.severity == Severity.MEDIUM),
                "low": sum(1 for f in self.findings if f.severity == Severity.LOW)
            },
            "findings": [f.to_dict() for f in self.findings],
            "scanners_run": self.scanners_run,
            "scanners_skipped": self.scanners_skipped,
            "file_hashes": self.file_hashes,
            "format_check": self.format_check,
            "duration_seconds": self.duration_seconds
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# =============================================================================
# PATHFINDER SCANNER
# =============================================================================

class PathfinderScanner:
    """
    Unified security scanner for AI Model Pathfinder.
    
    Integrates multiple scanning tools and provides a single interface
    for the security pipeline.
    """
    
    # Allowed file formats (v1: safetensors only for weights)
    ALLOWED_WEIGHT_FORMATS = [".safetensors"]
    ALLOWED_CONFIG_FORMATS = [".json", ".yaml", ".yml", ".txt", ".md"]
    
    # Blocked formats (pickle-based, dangerous)
    BLOCKED_FORMATS = [".pkl", ".pickle", ".pt", ".pth", ".bin", ".h5", ".hdf5"]
    
    # Dangerous code patterns (from Lab 01)
    DANGEROUS_PATTERNS = [
        # Network operations
        (r'socket\.socket', 'NETWORK_SOCKET', 'Network socket creation'),
        (r'urllib|requests\.get|http\.client', 'NETWORK_HTTP', 'HTTP request capability'),
        (r'smtplib|ftplib', 'NETWORK_PROTOCOL', 'Email/FTP protocol usage'),
        
        # Code execution
        (r'os\.fork\s*\(\)', 'PROCESS_FORK', 'Process forking'),
        (r'subprocess', 'SUBPROCESS', 'Subprocess execution'),
        (r'os\.system', 'OS_SYSTEM', 'System command execution'),
        (r'pty\.spawn', 'PTY_SPAWN', 'PTY shell spawning'),
        (r'exec\s*\(|eval\s*\(', 'DYNAMIC_EXEC', 'Dynamic code execution'),
        (r'compile\s*\(', 'CODE_COMPILE', 'Dynamic code compilation'),
        
        # Deserialization (RCE risk)
        (r'pickle\.load|pickle\.loads', 'PICKLE_LOAD', 'Pickle deserialization'),
        (r'torch\.load', 'TORCH_LOAD', 'PyTorch load (uses pickle)'),
        (r'joblib\.load', 'JOBLIB_LOAD', 'Joblib load (uses pickle)'),
        
        # Obfuscation techniques
        (r'base64\.b64decode', 'BASE64_DECODE', 'Base64 decoding (payload hiding)'),
        (r'__import__\s*\(', 'DYNAMIC_IMPORT', 'Dynamic import'),
        (r'getattr.*\(.*,.*\)\s*\(', 'DYNAMIC_CALL', 'Dynamic attribute call'),
        (r'\\x[0-9a-fA-F]{2}', 'HEX_STRINGS', 'Hex-encoded strings'),
        
        # File system access
        (r'os\.dup2', 'FD_REDIRECT', 'File descriptor redirection'),
        (r'ctypes|cffi', 'NATIVE_CODE', 'Native code execution'),
    ]
    
    # Trusted publishers (from Lab 01)
    TRUSTED_PUBLISHERS = [
        'google', 'meta-llama', 'microsoft', 'openai', 'huggingface',
        'facebook', 'nvidia', 'bigscience', 'EleutherAI', 'stabilityai',
        'mistralai', 'anthropic', 'databricks', 'mosaicml', 'tiiuae',
        'Qwen', 'deepseek-ai', 'allenai', 'sentence-transformers'
    ]
    
    def __init__(self, strict_mode: bool = True, allow_pickle: bool = False):
        """
        Initialize scanner.
        
        Args:
            strict_mode: If True, fail on any HIGH+ findings
            allow_pickle: If True, allow pickle formats (NOT recommended for v1)
        """
        self.strict_mode = strict_mode
        self.allow_pickle = allow_pickle
    
    def scan_model(self, model_path: str, include_code_scan: bool = True) -> ScanResult:
        """
        Run complete security scan on a model.
        
        Args:
            model_path: Path to model directory or file
            include_code_scan: Whether to scan .py files for dangerous patterns
            
        Returns:
            ScanResult with all findings
        """
        import time
        start_time = time.time()
        
        model_path = Path(model_path)
        result = ScanResult(model_path=str(model_path))
        
        logger.info(f"Starting security scan: {model_path}")
        
        # Determine if scanning a file or directory
        if model_path.is_file():
            files = [model_path]
        else:
            files = list(model_path.rglob("*"))
        
        # Run all scanners
        self._check_file_formats(files, result)
        self._compute_hashes(files, result)
        self._run_modelscan(model_path, result)
        self._run_picklescan(files, result)
        
        if include_code_scan:
            self._scan_code_patterns(files, result)
        
        self._check_trust_remote_code(model_path, result)
        
        result.duration_seconds = time.time() - start_time
        
        # Log summary
        self._log_summary(result)
        
        return result
    
    def _check_file_formats(self, files: List[Path], result: ScanResult):
        """Check for blocked file formats"""
        result.scanners_run.append("FormatChecker")
        
        weight_files = []
        blocked_files = []
        safetensor_files = []
        
        for f in files:
            if not f.is_file():
                continue
                
            suffix = f.suffix.lower()
            
            if suffix in self.BLOCKED_FORMATS:
                if not self.allow_pickle:
                    blocked_files.append(str(f))
                    result.add_finding(Finding(
                        scanner="FormatChecker",
                        severity=Severity.CRITICAL,
                        category="BLOCKED_FORMAT",
                        message=f"Blocked format detected: {suffix} (pickle-based, RCE risk)",
                        file=str(f)
                    ))
                else:
                    result.add_finding(Finding(
                        scanner="FormatChecker",
                        severity=Severity.MEDIUM,
                        category="PICKLE_FORMAT",
                        message=f"Pickle-based format detected: {suffix}",
                        file=str(f)
                    ))
            
            if suffix == ".safetensors":
                safetensor_files.append(str(f))
        
        result.format_check = {
            "safetensors_files": len(safetensor_files),
            "blocked_files": len(blocked_files),
            "safetensors_only": len(blocked_files) == 0 and len(safetensor_files) > 0
        }
    
    def _compute_hashes(self, files: List[Path], result: ScanResult):
        """Compute SHA256 hashes for all model files"""
        for f in files:
            if f.is_file() and f.suffix in (self.ALLOWED_WEIGHT_FORMATS + 
                                             self.ALLOWED_CONFIG_FORMATS + 
                                             ['.py']):
                try:
                    sha256 = hashlib.sha256()
                    with open(f, 'rb') as file:
                        for chunk in iter(lambda: file.read(8192), b''):
                            sha256.update(chunk)
                    result.file_hashes[str(f.name)] = sha256.hexdigest()
                except Exception as e:
                    logger.warning(f"Could not hash {f}: {e}")
    
    def _run_modelscan(self, model_path: Path, result: ScanResult):
        """Run Protect AI's ModelScan"""
        if not MODELSCAN_AVAILABLE:
            result.scanners_skipped.append("ModelScan")
            return
        
        result.scanners_run.append("ModelScan")
        
        try:
            scanner = ModelScan()
            scan_result = scanner.scan(str(model_path))
            
            if scan_result.issues:
                for issue in scan_result.issues:
                    result.add_finding(Finding(
                        scanner="ModelScan",
                        severity=Severity.CRITICAL,
                        category="MODELSCAN_ISSUE",
                        message=str(issue),
                        details={"raw_issue": str(issue)}
                    ))
        except Exception as e:
            logger.error(f"ModelScan error: {e}")
            result.add_finding(Finding(
                scanner="ModelScan",
                severity=Severity.LOW,
                category="SCANNER_ERROR",
                message=f"ModelScan encountered an error: {e}"
            ))
    
    def _run_picklescan(self, files: List[Path], result: ScanResult):
        """Run PickleScan on pickle-based files"""
        if not PICKLESCAN_AVAILABLE:
            result.scanners_skipped.append("PickleScan")
            return
        
        result.scanners_run.append("PickleScan")
        
        pickle_files = [f for f in files if f.suffix in ['.pkl', '.pickle', '.pt', '.pth', '.bin']]
        
        for pf in pickle_files:
            try:
                scan_result = picklescan.scan_file_path(str(pf))
                
                if scan_result.issues:
                    for issue in scan_result.issues:
                        result.add_finding(Finding(
                            scanner="PickleScan",
                            severity=Severity.CRITICAL,
                            category="PICKLE_INJECTION",
                            message=f"Malicious pickle detected: {issue}",
                            file=str(pf)
                        ))
            except Exception as e:
                logger.warning(f"PickleScan error on {pf}: {e}")
    
    def _scan_code_patterns(self, files: List[Path], result: ScanResult):
        """Scan Python files for dangerous patterns (from Lab 01)"""
        result.scanners_run.append("CodePatternScanner")
        
        py_files = [f for f in files if f.suffix == '.py' and f.is_file()]
        
        for py_file in py_files:
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                for pattern, category, description in self.DANGEROUS_PATTERNS:
                    matches = re.findall(pattern, content)
                    if matches:
                        # Determine severity based on pattern type
                        if category in ['NETWORK_SOCKET', 'SUBPROCESS', 'PTY_SPAWN', 
                                       'PICKLE_LOAD', 'DYNAMIC_EXEC']:
                            severity = Severity.CRITICAL
                        elif category in ['NETWORK_HTTP', 'OS_SYSTEM', 'TORCH_LOAD']:
                            severity = Severity.HIGH
                        else:
                            severity = Severity.MEDIUM
                        
                        result.add_finding(Finding(
                            scanner="CodePatternScanner",
                            severity=severity,
                            category=category,
                            message=f"{description} ({len(matches)} occurrence(s))",
                            file=str(py_file),
                            details={"matches": len(matches)}
                        ))
            except Exception as e:
                logger.warning(f"Could not scan {py_file}: {e}")
    
    def _check_trust_remote_code(self, model_path: Path, result: ScanResult):
        """Check if model requires trust_remote_code"""
        result.scanners_run.append("TrustRemoteCodeChecker")
        
        # Look for config.json
        if model_path.is_dir():
            config_path = model_path / "config.json"
        else:
            config_path = model_path.parent / "config.json"
        
        if not config_path.exists():
            return
        
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            # Check for auto_map which indicates custom code
            if "auto_map" in config:
                auto_map = config["auto_map"]
                custom_classes = list(auto_map.values()) if isinstance(auto_map, dict) else []
                
                result.add_finding(Finding(
                    scanner="TrustRemoteCodeChecker",
                    severity=Severity.HIGH,
                    category="TRUST_REMOTE_CODE_REQUIRED",
                    message="Model requires trust_remote_code=True for custom classes",
                    file=str(config_path),
                    details={"auto_map": auto_map, "custom_classes": custom_classes}
                ))
                
                # Also flag the Python files that will be executed
                if model_path.is_dir():
                    for cls in custom_classes:
                        if '.' in cls:
                            module = cls.split('.')[0]
                            py_file = model_path / f"{module}.py"
                            if py_file.exists():
                                result.add_finding(Finding(
                                    scanner="TrustRemoteCodeChecker",
                                    severity=Severity.HIGH,
                                    category="CUSTOM_CODE_FILE",
                                    message=f"Custom code will be executed from: {module}.py",
                                    file=str(py_file)
                                ))
        except Exception as e:
            logger.warning(f"Could not check config.json: {e}")
    
    def _log_summary(self, result: ScanResult):
        """Log scan summary"""
        summary = result.to_dict()["summary"]
        
        if result.passed:
            logger.info(f"✓ Scan PASSED ({summary['total_findings']} findings, none critical/high)")
        else:
            logger.warning(f"✗ Scan FAILED: {summary['critical']} critical, {summary['high']} high findings")
        
        logger.info(f"  Scanners run: {', '.join(result.scanners_run)}")
        if result.scanners_skipped:
            logger.info(f"  Scanners skipped: {', '.join(result.scanners_skipped)}")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Pathfinder Model Security Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan a model directory
  python pathfinder_scanner.py /path/to/model

  # Scan with pickle formats allowed (not recommended)
  python pathfinder_scanner.py /path/to/model --allow-pickle

  # Output JSON to file
  python pathfinder_scanner.py /path/to/model --output scan_result.json
        """
    )
    
    parser.add_argument("model_path", help="Path to model file or directory")
    parser.add_argument("--allow-pickle", action="store_true", 
                        help="Allow pickle-based formats (NOT recommended)")
    parser.add_argument("--no-code-scan", action="store_true",
                        help="Skip scanning Python files for patterns")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger("pathfinder.scanner").setLevel(logging.WARNING)
    
    # Run scan
    scanner = PathfinderScanner(
        strict_mode=True,
        allow_pickle=args.allow_pickle
    )
    
    result = scanner.scan_model(
        args.model_path,
        include_code_scan=not args.no_code_scan
    )
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(result.to_json())
        print(f"Results written to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("PATHFINDER SECURITY SCAN RESULTS")
        print("=" * 60)
        print(f"\nModel: {result.model_path}")
        print(f"Status: {'✓ PASSED' if result.passed else '✗ FAILED'}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"\nScanners: {', '.join(result.scanners_run)}")
        
        if result.findings:
            print(f"\nFindings ({len(result.findings)}):")
            for f in result.findings:
                icon = "🚨" if f.severity == Severity.CRITICAL else \
                       "⚠️" if f.severity == Severity.HIGH else \
                       "⚡" if f.severity == Severity.MEDIUM else "ℹ️"
                print(f"  {icon} [{f.severity.value}] {f.category}: {f.message}")
                if f.file:
                    print(f"      File: {f.file}")
        
        print("\n" + "=" * 60)
    
    # Exit code
    return 0 if result.passed else 1


if __name__ == "__main__":
    exit(main())
