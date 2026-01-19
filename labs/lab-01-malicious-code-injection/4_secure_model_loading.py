#!/usr/bin/env python3
"""
Step 4: Secure Model Loading with Industry Defenses

Defense layers:
1. ModelScan (Protect AI) - Scans for malicious code in model files
2. SafeTensors check - Prefer safe formats over pickle/H5
3. Hash verification - Compare against known-good hash
4. Safe mode enforcement - Block unsafe deserialization

Run AFTER 2_inject_malicious_code.py to see detection in action.

Author: GopeshK
License: MIT License
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("absl").setLevel(logging.ERROR)

import hashlib
import json
import re
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime

# ModelScan (Protect AI) - Industry standard
try:
    from modelscan.modelscan import ModelScan
    MODELSCAN_AVAILABLE = True
except ImportError:
    MODELSCAN_AVAILABLE = False

MODEL_PATH = "model.h5"
HASH_REGISTRY = "model_hashes.json"

# =============================================================================
# SECURITY LAYER 1: ModelScan (Protect AI)
# =============================================================================

class ModelSecurityScanner:
    """
    Scan ML models for security issues using multiple detection methods.
    Inspired by Protect AI's ModelScan.
    """
    
    # Dangerous patterns in serialized models
    DANGEROUS_PATTERNS = [
        # Python code execution
        r'__reduce__',
        r'__reduce_ex__',
        r'exec\s*\(',
        r'eval\s*\(',
        r'compile\s*\(',
        r'__import__',
        r'importlib',
        r'subprocess',
        r'os\.system',
        r'os\.popen',
        r'commands\.',
        
        # Network operations
        r'socket\.',
        r'urllib',
        r'requests\.',
        r'smtplib',
        r'ftplib',
        
        # File operations (suspicious in model context)
        r'open\s*\([^)]*["\']w',  # Write mode
        r'shutil\.',
        
        # Shell commands
        r'/bin/sh',
        r'/bin/bash',
        r'cmd\.exe',
        r'powershell',
    ]
    
    # Known malicious layer names
    SUSPICIOUS_LAYER_NAMES = [
        'malicious', 'payload', 'backdoor', 'exploit', 
        'shell', 'reverse', 'inject', 'hack'
    ]
    
    def __init__(self):
        self.findings = []
        
    def scan(self, model_path: str) -> dict:
        """Comprehensive security scan of model file"""
        self.findings = []
        
        results = {
            "file": model_path,
            "timestamp": datetime.now().isoformat(),
            "safe": True,
            "findings": [],
            "checks_performed": []
        }
        
        # Check 1: Use ModelScan if available (industry standard)
        if MODELSCAN_AVAILABLE:
            results["checks_performed"].append("ModelScan (Protect AI)")
            try:
                scanner = ModelScan()
                scan_results = scanner.scan(model_path)
                if scan_results.issues:
                    results["safe"] = False
                    for issue in scan_results.issues:
                        results["findings"].append({
                            "type": "MODELSCAN",
                            "severity": "CRITICAL",
                            "message": str(issue)
                        })
            except Exception as e:
                results["findings"].append({
                    "type": "SCAN_ERROR",
                    "severity": "WARNING",
                    "message": f"ModelScan error: {e}"
                })
        
        # Check 2: H5 file structure analysis
        results["checks_performed"].append("H5 Structure Analysis")
        h5_findings = self._scan_h5_structure(model_path)
        if h5_findings:
            results["safe"] = False
            results["findings"].extend(h5_findings)
        
        # Check 3: Lambda/Custom layer detection
        results["checks_performed"].append("Lambda Layer Detection")
        lambda_findings = self._detect_lambda_layers(model_path)
        if lambda_findings:
            results["safe"] = False
            results["findings"].extend(lambda_findings)
        
        # Check 4: Suspicious pattern scan
        results["checks_performed"].append("Pattern Matching")
        pattern_findings = self._scan_patterns(model_path)
        if pattern_findings:
            results["safe"] = False
            results["findings"].extend(pattern_findings)
        
        return results
    
    def _scan_h5_structure(self, model_path: str) -> list:
        """Analyze H5 file structure for anomalies"""
        findings = []
        try:
            with h5py.File(model_path, 'r') as f:
                # Check for suspicious groups/datasets
                def check_group(name, obj):
                    # Large string datasets might contain code
                    if isinstance(obj, h5py.Dataset):
                        if obj.dtype.kind == 'S' or obj.dtype.kind == 'O':
                            try:
                                data = obj[()]
                                if isinstance(data, bytes):
                                    data = data.decode('utf-8', errors='ignore')
                                elif isinstance(data, np.ndarray):
                                    data = str(data)
                                
                                # Check for dangerous patterns in string data
                                for pattern in self.DANGEROUS_PATTERNS[:5]:  # Check top dangerous patterns
                                    if re.search(pattern, str(data), re.IGNORECASE):
                                        findings.append({
                                            "type": "H5_SUSPICIOUS_DATA",
                                            "severity": "HIGH",
                                            "message": f"Suspicious pattern '{pattern}' in {name}"
                                        })
                            except:
                                pass
                
                f.visititems(check_group)
        except Exception as e:
            findings.append({
                "type": "H5_ERROR",
                "severity": "WARNING", 
                "message": f"Could not analyze H5 structure: {e}"
            })
        return findings
    
    def _detect_lambda_layers(self, model_path: str) -> list:
        """Detect Lambda layers which can execute arbitrary code"""
        findings = []
        try:
            with h5py.File(model_path, 'r') as f:
                if 'model_config' in f.attrs:
                    config = f.attrs['model_config']
                    if isinstance(config, bytes):
                        config = config.decode('utf-8')
                    
                    # Parse model config
                    try:
                        config_dict = json.loads(config)
                        layers = self._extract_layers(config_dict)
                        
                        for layer in layers:
                            layer_class = layer.get('class_name', '')
                            layer_name = layer.get('config', {}).get('name', '')
                            
                            # Lambda layers are dangerous
                            if layer_class == 'Lambda':
                                findings.append({
                                    "type": "LAMBDA_LAYER",
                                    "severity": "CRITICAL",
                                    "message": f"Lambda layer detected: '{layer_name}' - can execute arbitrary code"
                                })
                            
                            # Check for suspicious names
                            for suspicious in self.SUSPICIOUS_LAYER_NAMES:
                                if suspicious in layer_name.lower():
                                    findings.append({
                                        "type": "SUSPICIOUS_LAYER_NAME",
                                        "severity": "HIGH",
                                        "message": f"Suspicious layer name: '{layer_name}'"
                                    })
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            pass
        return findings
    
    def _extract_layers(self, config: dict) -> list:
        """Recursively extract layer configs from model config"""
        layers = []
        if isinstance(config, dict):
            if 'class_name' in config:
                layers.append(config)
            if 'config' in config and isinstance(config['config'], dict):
                if 'layers' in config['config']:
                    for layer in config['config']['layers']:
                        layers.extend(self._extract_layers(layer))
        return layers
    
    def _scan_patterns(self, model_path: str) -> list:
        """Scan raw file for dangerous patterns"""
        findings = []
        try:
            with open(model_path, 'rb') as f:
                content = f.read()
                content_str = content.decode('utf-8', errors='ignore')
                
                for pattern in self.DANGEROUS_PATTERNS:
                    matches = re.findall(pattern, content_str, re.IGNORECASE)
                    if matches:
                        findings.append({
                            "type": "DANGEROUS_PATTERN",
                            "severity": "HIGH",
                            "message": f"Found '{pattern}' pattern ({len(matches)} occurrences)"
                        })
        except Exception as e:
            pass
        return findings


# =============================================================================
# SECURITY LAYER 2: Hash Verification
# =============================================================================

def compute_hash(file_path: str) -> str:
    """Compute SHA256 hash of file"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def verify_hash(file_path: str, expected_hash: str) -> bool:
    """Verify file hash matches expected value"""
    actual = compute_hash(file_path)
    return actual == expected_hash

def load_hash_registry() -> dict:
    """Load known-good hashes from registry"""
    if os.path.exists(HASH_REGISTRY):
        with open(HASH_REGISTRY, 'r') as f:
            return json.load(f)
    return {}

def save_hash_registry(registry: dict):
    """Save hash registry"""
    with open(HASH_REGISTRY, 'w') as f:
        json.dump(registry, f, indent=2)


# =============================================================================
# SECURITY LAYER 3: Safe Model Loading
# =============================================================================

def secure_load_model(model_path: str, skip_scan: bool = False):
    """
    Securely load a model with all defense layers.
    Runs ALL checks and reports ALL failures before blocking.
    
    Defense layers:
    1. Security scan (ModelScan + custom checks)
    2. Hash verification (if registered)
    3. Safe mode loading (no unsafe deserialization)
    """
    import tensorflow as tf
    import joblib
    
    print("\n" + "=" * 60)
    print("  SECURE MODEL LOADER")
    print("  Industry-Standard Security Checks")
    print("=" * 60)
    
    all_failures = []  # Collect all failures
    
    # =========================================================================
    # Layer 1: Security Scan
    # =========================================================================
    scan_passed = True
    scan_findings = []
    
    if not skip_scan:
        print(f"\n[1/3] 🔍 Scanning model for security issues...")
        scanner = ModelSecurityScanner()
        results = scanner.scan(model_path)
        
        print(f"      Checks: {', '.join(results['checks_performed'])}")
        
        if not results["safe"]:
            scan_passed = False
            scan_findings = results["findings"]
            print(f"      ❌ FAILED - {len(scan_findings)} security issues found")
            for finding in scan_findings:
                severity = finding["severity"]
                icon = "🔴" if severity == "CRITICAL" else "🟠" if severity == "HIGH" else "🟡"
                print(f"         {icon} [{severity}] {finding['type']}: {finding['message']}")
            all_failures.append(("Security Scan", scan_findings))
        else:
            print(f"      ✅ PASSED - No security issues detected")
    
    # =========================================================================
    # Layer 2: Hash Verification
    # =========================================================================
    hash_passed = True
    
    print(f"\n[2/3] 🔐 Verifying model integrity (hash check)...")
    registry = load_hash_registry()
    current_hash = compute_hash(model_path)
    
    if model_path in registry:
        if registry[model_path] == current_hash:
            print(f"      ✅ PASSED - Hash matches registered value")
        else:
            hash_passed = False
            print(f"      ❌ FAILED - Hash MISMATCH (model tampered!)")
            print(f"         Expected: {registry[model_path][:32]}...")
            print(f"         Actual:   {current_hash[:32]}...")
            all_failures.append(("Hash Verification", [{"message": "Hash mismatch - model modified"}]))
    else:
        print(f"      ⚠️  SKIPPED - No registered hash (first load)")
        print(f"         Current: {current_hash[:32]}...")
    
    # =========================================================================
    # Layer 3: Safe Mode Loading Test
    # =========================================================================
    safe_mode_passed = True
    
    print(f"\n[3/3] 📦 Testing safe_mode=True loading...")
    
    try:
        # Attempt safe loading - don't keep the model, just test
        test_model = tf.keras.models.load_model(model_path, safe_mode=True)
        del test_model  # Clean up
        print(f"      ✅ PASSED - Model can be loaded safely")
    except Exception as e:
        safe_mode_passed = False
        error_msg = str(e)
        print(f"      ❌ FAILED - Model contains unsafe code!")
        print(f"         Error: {error_msg[:100]}...")
        all_failures.append(("Safe Mode Loading", [{"message": f"Blocked: {error_msg[:80]}"}]))
    
    # =========================================================================
    # Final Decision
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("  SECURITY ASSESSMENT SUMMARY")
    print(f"{'=' * 60}")
    
    print(f"\n  Layer 1 - Security Scan:     {'❌ FAILED' if not scan_passed else '✅ PASSED'}")
    print(f"  Layer 2 - Hash Verification: {'❌ FAILED' if not hash_passed else '✅ PASSED' if model_path in registry else '⚠️  SKIPPED'}")
    print(f"  Layer 3 - Safe Mode Loading: {'❌ FAILED' if not safe_mode_passed else '✅ PASSED'}")
    
    if all_failures:
        print(f"\n{'=' * 60}")
        print(f"  ⛔ MODEL BLOCKED - {len(all_failures)} CHECK(S) FAILED")
        print(f"{'=' * 60}")
        
        print("\n  DETAILED FINDINGS:\n")
        for check_name, findings in all_failures:
            print(f"  ── {check_name} ──")
            for finding in findings:
                msg = finding.get('message', str(finding))
                print(f"     • {msg}")
            print()
        
        print(f"{'=' * 60}")
        print("  RECOMMENDED ACTIONS:")
        print("  1. Do NOT load this model")
        print("  2. Quarantine the model file")
        print("  3. Report to security team")
        print("  4. Obtain model from trusted source")
        print(f"{'=' * 60}\n")
        
        raise SecurityError(f"Model failed {len(all_failures)} security check(s)")
    
    # All checks passed - load the model for real
    print(f"\n{'=' * 60}")
    print(f"  ✅ ALL CHECKS PASSED - Loading model...")
    print(f"{'=' * 60}\n")
    
    model = tf.keras.models.load_model(model_path, safe_mode=True)
    
    return model
    
    return model


class SecurityError(Exception):
    """Raised when a security check fails"""
    pass


# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  Lab 01: Secure Model Loading Demo")
    print("  Testing Industry Defense Mechanisms")
    print("=" * 60)
    
    if MODELSCAN_AVAILABLE:
        print("\n✅ ModelScan (Protect AI) is installed")
    else:
        print("\n⚠️  ModelScan not installed - using fallback scanner")
        print("   Install: pip install modelscan")
    
    print(f"\nTarget model: {MODEL_PATH}")
    
    try:
        # Try secure loading
        model = secure_load_model(MODEL_PATH)
        
        # If we get here, model is safe - run inference
        import joblib
        vectorizer = joblib.load("vectorizer.joblib")
        
        with open("responses.json", "r") as f:
            responses = json.load(f)
        
        print("\n[*] Model loaded successfully. Ready for inference.")
        prompt = input("\nAsk a question: ")
        X_input = vectorizer.transform([prompt]).toarray()
        prediction = model.predict(X_input, verbose=0)
        predicted_label = np.argmax(prediction)
        print(f"\nResponse: {responses[predicted_label]}")
        
    except SecurityError as e:
        print(f"\n❌ Security Error: {e}")
        print("\n[!] Model was NOT loaded due to security concerns.")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
