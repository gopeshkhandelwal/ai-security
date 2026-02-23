# NIST RMF Artifacts for Lab 05: Malicious Code Injection Defense

**System Name:** Secure ML Model Loading Pipeline  
**System Owner:** AI Platform Security Team  
**Date:** February 23, 2026  
**Classification:** High Impact (FIPS 199)

---

## NIST AI Risk Management Framework (AI RMF)

### GOVERN: AI Governance and Accountability

**AI Governance Policy Statement:**  
All ML models must be scanned for malicious code before deployment. Models using unsafe serialization formats (Pickle, HDF5 Lambda layers) must undergo additional security review. Only approved model formats (SafeTensors, ONNX) are permitted in production without explicit security waiver.

**Roles and Responsibilities:**

| Role | Responsibility | Accountable Person |
|------|---------------|-------------------|
| AI System Owner | Model loading security | Director, AI Platform |
| Security Lead | Malware scanning policy | Chief Security Officer |
| ML Engineer | Safe model deployment | ML Engineering Lead |
| DevSecOps Lead | CI/CD integration | Platform Engineering Lead |

**AI Risk Tolerance:**

| Risk Category | Tolerance Level | Rationale |
|---------------|-----------------|-----------|
| Arbitrary Code Execution | Zero Tolerance | Full system compromise |
| Reverse Shell | Zero Tolerance | Remote access |
| Data Exfiltration | Zero Tolerance | Data breach |
| Cryptominer | Very Low Tolerance | Resource abuse |

---

### MAP: AI Risk Identification

**System Context:**  
ML model files can contain executable code that runs during model loading. Attackers inject malicious payloads into model weights, Lambda layers, or pickle serialization to achieve code execution on victim systems.

**Threat Actors:**

| Actor | Motivation | Capability | Likelihood |
|-------|------------|------------|------------|
| APT | Persistent access | Very High | Medium |
| Attacker | System compromise | High | High |
| Supply Chain | Mass exploitation | High | Medium |
| Insider | Sabotage | High | Low |

**AI-Specific Risk Catalog:**

| Risk ID | Risk Description | MITRE ATLAS | Impact | Likelihood |
|---------|------------------|-------------|--------|------------|
| MC-R001 | Malicious Keras Lambda layer | AML.T0010 | Critical | High |
| MC-R002 | Pickle code execution | AML.T0010 | Critical | High |
| MC-R003 | PyTorch model backdoor | AML.T0010 | Critical | Medium |
| MC-R004 | SavedModel graph injection | AML.T0010 | Critical | Medium |
| MC-R005 | Reverse shell payload | AML.T0010 | Critical | Medium |

---

### MEASURE: Risk Assessment and Verification

**Risk Measurement Methodology:**

| Measurement Type | Method | Frequency | Tool |
|------------------|--------|-----------|------|
| Malware Detection | Automated scanning | Every load | ModelScan |
| Pattern Coverage | Signature updates | Weekly | Security team |
| False Positive Rate | Manual review | Monthly | ML team |
| Evasion Testing | Red team | Quarterly | Security research |

**Protection Metrics:**

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Malware detection rate | >99% | 99.5% | ✅ Met |
| False positive rate | <1% | 0.3% | ✅ Met |
| Scan time (100MB model) | <5s | 1.2s | ✅ Met |
| CI/CD integration | 100% | 100% | ✅ Met |

---

### MANAGE: Risk Treatment and Response

**Risk Treatment Decisions:**

| Risk ID | Treatment | Control | Residual Risk |
|---------|-----------|---------|---------------|
| MC-R001 | Mitigate | ModelScan + safe_mode | Very Low |
| MC-R002 | Mitigate | Pickle scanning | Very Low |
| MC-R003 | Mitigate | weights_only=True | Very Low |
| MC-R004 | Mitigate | Graph analysis | Low |
| MC-R005 | Mitigate | Network monitoring | Very Low |

**Control Implementation:**

| Control Type | Implementation | Status |
|--------------|----------------|--------|
| **Preventive** | ModelScan in CI/CD | ✅ Active |
| **Preventive** | Keras safe_mode=True | ✅ Active |
| **Preventive** | PyTorch weights_only=True | ✅ Active |
| **Detective** | Runtime behavior monitoring | ✅ Active |
| **Corrective** | Automatic quarantine | ✅ Active |

---

## NIST 800-53 Control Mapping

| Control ID | Control Name | Implementation | Status |
|------------|--------------|----------------|--------|
| **SI-3** | Malicious Code Protection | ModelScan integration | ✅ Implemented |
| **SI-7** | Software Integrity | Hash verification | ✅ Implemented |
| **CM-7** | Least Functionality | Safe loading modes | ✅ Implemented |
| **SA-12** | Supply Chain Protection | Model provenance | ✅ Implemented |
| **AU-6** | Audit Analysis | Scan result logging | ✅ Implemented |

---

## Malicious Patterns Detected

| Pattern | Description | Detection Method |
|---------|-------------|------------------|
| Lambda layer | Arbitrary code in Keras | AST analysis |
| Pickle exec | Code in __reduce__ | Pickle scanner |
| Network calls | socket/requests | Import analysis |
| File operations | os.system/subprocess | String scanning |
| Base64 payloads | Encoded malware | Entropy analysis |

---

## Security Assessment Results

| Test Case | Expected Result | Actual Result | Status |
|-----------|-----------------|---------------|--------|
| Keras model with malicious Lambda | Blocked | Detected: UNSAFE | ✅ Pass |
| Pickle with reverse shell | Blocked | Detected: UNSAFE | ✅ Pass |
| Clean model with safe_mode | Loaded | Loaded successfully | ✅ Pass |
| SafeTensors clean model | Loaded | Loaded successfully | ✅ Pass |

---

## Safe Loading Patterns

| Framework | Unsafe Method | Safe Alternative |
|-----------|--------------|------------------|
| Keras | `load_model()` | `load_model(safe_mode=True)` |
| PyTorch | `torch.load()` | `torch.load(weights_only=True)` |
| Pickle | `pickle.load()` | SafeTensors or verified sources |
| General | HuggingFace default | SafeTensors format |

---

## Appendix: MITRE ATLAS Mapping

| Technique ID | Technique Name | Lab 05 Mitigation |
|--------------|----------------|-------------------|
| **AML.T0010** | ML Supply Chain Compromise | ModelScan + safe loading |
| **AML.T0011** | Publish Poisoned Model | Model provenance verification |

---

*Document Version: 1.0*  
*Last Updated: February 23, 2026*
