#!/usr/bin/env python3
"""
Pathfinder Security Scanner
Scans models for security issues before deployment.
"""
import re
import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from enum import Enum
from datetime import datetime


class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class Finding:
    scanner: str
    severity: str
    category: str
    message: str
    file: Optional[str] = None


@dataclass 
class ScanResult:
    model_path: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    passed: bool = True
    findings: List[Finding] = field(default_factory=list)


# Dangerous code patterns to detect
DANGEROUS_PATTERNS = [
    (r'socket\.socket', 'NETWORK_SOCKET', 'Network socket creation', Severity.CRITICAL),
    (r'os\.fork\s*\(\)', 'PROCESS_FORK', 'Process forking', Severity.HIGH),
    (r'subprocess', 'SUBPROCESS', 'Subprocess execution', Severity.CRITICAL),
    (r'pty\.spawn', 'PTY_SPAWN', 'PTY shell spawning', Severity.CRITICAL),
    (r'os\.dup2', 'FD_REDIRECT', 'File descriptor redirection', Severity.HIGH),
    (r'exec\s*\(|eval\s*\(', 'DYNAMIC_EXEC', 'Dynamic code execution', Severity.HIGH),
    (r'os\.system', 'OS_SYSTEM', 'System command execution', Severity.CRITICAL),
    (r'pickle\.load', 'PICKLE_LOAD', 'Pickle deserialization', Severity.HIGH),
    (r'torch\.load', 'TORCH_LOAD', 'PyTorch load (uses pickle)', Severity.MEDIUM),
    (r'__reduce__', 'PICKLE_REDUCE', 'Pickle reduce method (RCE risk)', Severity.CRITICAL),
    (r'requests\.(get|post|put)', 'HTTP_REQUEST', 'HTTP request', Severity.MEDIUM),
    (r'urllib', 'URLLIB', 'URL library usage', Severity.LOW),
]

# Blocked file formats (pickle-based)
BLOCKED_FORMATS = ['.pkl', '.pickle', '.pt', '.pth']

# Warning formats (need review)
WARNING_FORMATS = ['.bin']


def scan_file_formats(model_path: Path, allow_pickle: bool = False) -> List[Finding]:
    """Check for blocked file formats."""
    findings = []
    
    for f in model_path.rglob("*"):
        if not f.is_file():
            continue
            
        if f.suffix in BLOCKED_FORMATS and not allow_pickle:
            findings.append(Finding(
                scanner="FormatChecker",
                severity=Severity.CRITICAL.value,
                category="BLOCKED_FORMAT",
                message=f"Blocked pickle-based format: {f.suffix}",
                file=str(f.name)
            ))
        elif f.suffix in WARNING_FORMATS:
            findings.append(Finding(
                scanner="FormatChecker",
                severity=Severity.MEDIUM.value,
                category="WARNING_FORMAT",
                message=f"Format may contain pickle: {f.suffix}",
                file=str(f.name)
            ))
    
    return findings


def scan_python_files(model_path: Path) -> List[Finding]:
    """Scan Python files for dangerous patterns."""
    findings = []
    
    for py_file in model_path.rglob("*.py"):
        try:
            content = py_file.read_text(errors='ignore')
            
            for pattern, category, desc, severity in DANGEROUS_PATTERNS:
                if re.search(pattern, content):
                    findings.append(Finding(
                        scanner="CodePatternScanner",
                        severity=severity.value,
                        category=category,
                        message=desc,
                        file=str(py_file.name)
                    ))
        except Exception:
            pass
    
    return findings


def check_trust_remote_code(model_path: Path) -> List[Finding]:
    """Check if model requires trust_remote_code."""
    findings = []
    config_path = model_path / "config.json"
    
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            if "auto_map" in config:
                findings.append(Finding(
                    scanner="TrustRemoteCodeChecker",
                    severity=Severity.HIGH.value,
                    category="TRUST_REMOTE_CODE_REQUIRED",
                    message="Model requires trust_remote_code=True (custom code)",
                    file="config.json"
                ))
        except Exception:
            pass
    
    return findings


def run_modelscan(model_path: Path) -> List[Finding]:
    """Run ModelScan if available."""
    findings = []
    
    try:
        from modelscan.modelscan import ModelScan
        scanner = ModelScan()
        results = scanner.scan(str(model_path))
        
        for issue in results.issues.all_issues:
            findings.append(Finding(
                scanner="ModelScan",
                severity=issue.severity.name,
                category=issue.code,
                message=issue.description,
                file=str(issue.source) if hasattr(issue, 'source') else None
            ))
    except ImportError:
        pass  # ModelScan not installed
    except Exception as e:
        findings.append(Finding(
            scanner="ModelScan",
            severity=Severity.LOW.value,
            category="SCAN_ERROR",
            message=f"ModelScan error: {str(e)}",
            file=None
        ))
    
    return findings


def scan_model(model_path: str, allow_pickle: bool = False, 
               skip_modelscan: bool = False) -> ScanResult:
    """Run all security scans on a model."""
    model_path = Path(model_path)
    
    if not model_path.exists():
        return ScanResult(
            model_path=str(model_path),
            passed=False,
            findings=[Finding(
                scanner="PathValidator",
                severity=Severity.CRITICAL.value,
                category="PATH_NOT_FOUND",
                message=f"Model path does not exist: {model_path}",
                file=None
            )]
        )
    
    result = ScanResult(model_path=str(model_path))
    
    # Run all scanners
    result.findings.extend(scan_file_formats(model_path, allow_pickle))
    result.findings.extend(scan_python_files(model_path))
    result.findings.extend(check_trust_remote_code(model_path))
    
    if not skip_modelscan:
        result.findings.extend(run_modelscan(model_path))
    
    # Determine pass/fail based on severity
    critical_or_high = [f for f in result.findings 
                        if f.severity in ["CRITICAL", "HIGH"]]
    result.passed = len(critical_or_high) == 0
    
    return result


def print_findings(result: ScanResult):
    """Pretty print scan findings."""
    if result.passed:
        print(f"\n✅ SCAN PASSED: {result.model_path}")
    else:
        print(f"\n❌ SCAN FAILED: {result.model_path}")
    
    print(f"   Timestamp: {result.timestamp}")
    print(f"   Findings: {len(result.findings)}")
    
    if result.findings:
        print("\n   Issues:")
        for f in result.findings:
            icon = "🚨" if f.severity == "CRITICAL" else "⚠️" if f.severity == "HIGH" else "ℹ️"
            print(f"   {icon} [{f.severity}] {f.category}")
            print(f"      {f.message}")
            if f.file:
                print(f"      File: {f.file}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Scan model for security issues")
    parser.add_argument("model_path", help="Path to model directory")
    parser.add_argument("--allow-pickle", action="store_true", 
                        help="Allow pickle-based formats")
    parser.add_argument("--skip-modelscan", action="store_true",
                        help="Skip ModelScan integration")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--output", "-o", help="Save results to file")
    
    args = parser.parse_args()
    
    result = scan_model(args.model_path, args.allow_pickle, args.skip_modelscan)
    
    # Convert to dict for JSON
    result_dict = {
        "model_path": result.model_path,
        "timestamp": result.timestamp,
        "passed": result.passed,
        "findings": [asdict(f) for f in result.findings]
    }
    
    if args.json:
        print(json.dumps(result_dict, indent=2))
    else:
        print_findings(result)
    
    if args.output:
        Path(args.output).write_text(json.dumps(result_dict, indent=2))
        print(f"Results saved to: {args.output}")
    
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
