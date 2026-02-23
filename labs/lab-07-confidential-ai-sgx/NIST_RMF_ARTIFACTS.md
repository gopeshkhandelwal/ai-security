# NIST RMF Artifacts for Lab 07: Confidential AI with Intel SGX

**System Name:** Intel SGX Protected ML Inference System  
**System Owner:** AI Platform Security Team  
**Date:** February 23, 2026  
**Classification:** High Impact (FIPS 199)

---

## NIST AI Risk Management Framework (AI RMF)

### GOVERN: AI Governance and Accountability

**AI Governance Policy Statement:**  
Production ML inference systems processing sensitive data must run within Intel SGX enclaves. Model weights and inference data must be protected from privileged software attacks. Only attestation-verified enclaves may process confidential data.

**Roles and Responsibilities:**

| Role | Responsibility | Accountable Person |
|------|---------------|-------------------|
| AI System Owner | Confidential AI policy | Director, AI Platform |
| Enclave Security Lead | SGX deployment | Security Infrastructure Lead |
| Attestation Admin | IAS/DCAP configuration | Trust Services Lead |
| ML Engineer | Model enclave integration | ML Engineering Lead |

**AI Risk Tolerance:**

| Risk Category | Tolerance Level | Rationale |
|---------------|-----------------|-----------|
| Model Weight Extraction | Zero Tolerance | IP theft |
| Inference Data Leakage | Zero Tolerance | Privacy violation |
| Side-Channel Attack | Low Tolerance | Advanced threat |
| Enclave Compromise | Zero Tolerance | Trust violation |

---

### MAP: AI Risk Identification

**System Context:**  
SGX creates hardware-protected enclaves where code and data are encrypted in memory, protected even from root/hypervisor access. ML inference within enclaves protects model IP and user data from privileged attackers.

**Threat Actors:**

| Actor | Motivation | Capability | Likelihood |
|-------|------------|------------|------------|
| Malicious Admin | Data theft | Very High | Medium |
| Cloud Provider | Model inspection | Very High | Low |
| APT | IP theft | Very High | Medium |
| Researcher | Side-channel | Very High | Low |

**AI-Specific Risk Catalog:**

| Risk ID | Risk Description | MITRE ATLAS | Impact | Likelihood |
|---------|------------------|-------------|--------|------------|
| SG-R001 | Memory dump attack | AML.T0044 | Critical | High |
| SG-R002 | Privileged software attack | AML.T0044 | Critical | High |
| SG-R003 | Model weight extraction | AML.T0044 | Critical | Medium |
| SG-R004 | Side-channel attack | AML.T0044 | High | Low |
| SG-R005 | Attestation bypass | AML.T0044 | Critical | Low |

---

### MEASURE: Risk Assessment and Verification

**Risk Measurement Methodology:**

| Measurement Type | Method | Frequency | Tool |
|------------------|--------|-----------|------|
| Enclave Integrity | Remote attestation | Every request | IAS/DCAP |
| Memory Protection | Hardware verification | Continuous | SGX |
| Side-Channel Analysis | Security audit | Quarterly | Research team |
| Attestation Validation | Automated check | Every connection | Attestation service |

**Protection Metrics:**

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Memory encryption | 100% | 100% | ✅ Met |
| Attestation coverage | 100% | 100% | ✅ Met |
| Privileged attack protection | 100% | 100% | ✅ Met |
| Side-channel mitigations | Implemented | Active | ✅ Met |

---

### MANAGE: Risk Treatment and Response

**Risk Treatment Decisions:**

| Risk ID | Treatment | Control | Residual Risk |
|---------|-----------|---------|---------------|
| SG-R001 | Mitigate | SGX memory encryption | Very Low |
| SG-R002 | Mitigate | Enclave isolation | Very Low |
| SG-R003 | Mitigate | Model encryption at rest | Very Low |
| SG-R004 | Mitigate | Constant-time algorithms | Low |
| SG-R005 | Mitigate | Remote attestation | Very Low |

**Control Implementation:**

| Control Type | Implementation | Status |
|--------------|----------------|--------|
| **Preventive** | SGX enclave isolation | ✅ Active |
| **Preventive** | Memory Encryption Engine (MEE) | ✅ Active |
| **Preventive** | Gramine LibOS | ✅ Active |
| **Detective** | Intel Trust Authority | ✅ Active |
| **Corrective** | Attestation failure blocking | ✅ Active |

---

## NIST 800-53 Control Mapping

| Control ID | Control Name | Implementation | Status |
|------------|--------------|----------------|--------|
| **SC-28** | Protection of Information at Rest | SGX sealed storage | ✅ Implemented |
| **SC-12** | Cryptographic Key Management | SGX key hierarchy | ✅ Implemented |
| **SI-7** | Software Integrity | Enclave measurement | ✅ Implemented |
| **AU-10** | Non-repudiation | Remote attestation | ✅ Implemented |
| **SC-7** | Boundary Protection | Enclave isolation | ✅ Implemented |

---

## SGX Protection Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Application                       │
├─────────────────────────────────────────────────────┤
│                    Gramine LibOS                     │
├─────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────┐ │
│ │              SGX Enclave (EPC)                  │ │
│ │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │ │
│ │  │ Model   │  │ Input   │  │ Inference Code  │  │ │
│ │  │ Weights │  │ Data    │  │                 │  │ │
│ │  └─────────┘  └─────────┘  └─────────────────┘  │ │
│ └─────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────┤
│        Memory Encryption Engine (MEE)               │
├─────────────────────────────────────────────────────┤
│                  Intel SGX CPU                       │
└─────────────────────────────────────────────────────┘
```

---

## Security Assessment Results

| Test Case | Expected Result | Actual Result | Status |
|-----------|-----------------|---------------|--------|
| Memory dump as root | Encrypted | See garbage data | ✅ Pass |
| /proc/pid/mem access | Denied/Encrypted | Access blocked | ✅ Pass |
| Remote attestation | Valid quote | Quote verified | ✅ Pass |
| Model weight extraction | Protected | Weights encrypted | ✅ Pass |

---

## Intel SGX vs Standard Execution

| Aspect | Without SGX | With SGX |
|--------|-------------|----------|
| Memory visibility | Readable by root | Encrypted |
| Model weights | Extractable | Protected |
| Inference data | Exposed | Encrypted |
| Trust boundary | Process | CPU silicon |
| Attestation | None | Hardware-rooted |

---

## Appendix: MITRE ATLAS Mapping

| Technique ID | Technique Name | Lab 07 Mitigation |
|--------------|----------------|-------------------|
| **AML.T0044** | Full ML Model Access | SGX memory encryption |
| **AML.T0035** | ML Artifact Collection | Enclave isolation |
| **AML.T0024** | Exfiltration via Inference API | Sealed storage |

---

*Document Version: 1.0*  
*Last Updated: February 23, 2026*
