# NIST RMF Artifacts for Lab 08: TPM Attestation for AI Systems

**System Name:** TPM-Attested ML Inference Platform  
**System Owner:** AI Platform Security Team  
**Date:** February 23, 2026  
**Classification:** High Impact (FIPS 199)

---

## NIST AI Risk Management Framework (AI RMF)

### GOVERN: AI Governance and Accountability

**AI Governance Policy Statement:**  
All production ML inference servers must implement TPM 2.0 attestation. Platform integrity must be verified before model loading. Systems failing attestation must be automatically quarantined and investigated.

**Roles and Responsibilities:**

| Role | Responsibility | Accountable Person |
|------|---------------|-------------------|
| AI System Owner | Attestation policy | Director, AI Platform |
| TPM Administrator | TPM configuration | Security Infrastructure Lead |
| Trust Authority Admin | ITA deployment | Trust Services Lead |
| ML Operations | Platform health | ML Operations Lead |

**AI Risk Tolerance:**

| Risk Category | Tolerance Level | Rationale |
|---------------|-----------------|-----------|
| Compromised Platform | Zero Tolerance | Trust violation |
| Firmware Tampering | Zero Tolerance | Root of trust |
| Boot Process Attack | Zero Tolerance | System integrity |
| Attestation Bypass | Zero Tolerance | Trust anchor |

---

### MAP: AI Risk Identification

**System Context:**  
TPM 2.0 provides hardware-rooted attestation of platform integrity. PCR measurements capture firmware, bootloader, and OS state. Intel Trust Authority verifies attestation quotes before allowing model access.

**Threat Actors:**

| Actor | Motivation | Capability | Likelihood |
|-------|------------|------------|------------|
| APT | Persistent access | Very High | Medium |
| Firmware Attacker | Root compromise | Very High | Low |
| Insider | Infrastructure sabotage | High | Low |
| Supply Chain | Implant installation | Very High | Low |

**AI-Specific Risk Catalog:**

| Risk ID | Risk Description | MITRE ATLAS | Impact | Likelihood |
|---------|------------------|-------------|--------|------------|
| TP-R001 | Firmware rootkit | AML.T0044 | Critical | Low |
| TP-R002 | Bootloader compromise | AML.T0044 | Critical | Low |
| TP-R003 | OS kernel tampering | AML.T0044 | Critical | Medium |
| TP-R004 | Model loading on untrusted platform | AML.T0044 | Critical | Medium |
| TP-R005 | Cold boot attack | AML.T0044 | High | Low |

---

### MEASURE: Risk Assessment and Verification

**Risk Measurement Methodology:**

| Measurement Type | Method | Frequency | Tool |
|------------------|--------|-----------|------|
| Platform Attestation | TPM quote | Every boot | Intel Trust Authority |
| PCR Verification | Golden value comparison | Every attestation | ITA |
| Firmware Integrity | Measured boot | Every boot | TPM 2.0 |
| Log Analysis | Event log review | Continuous | Security team |

**Protection Metrics:**

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Attestation coverage | 100% | 100% | ✅ Met |
| PCR measurement accuracy | 100% | 100% | ✅ Met |
| Tamper detection rate | 100% | 100% | ✅ Met |
| Mean time to detect | <1 min | 30s | ✅ Met |

---

### MANAGE: Risk Treatment and Response

**Risk Treatment Decisions:**

| Risk ID | Treatment | Control | Residual Risk |
|---------|-----------|---------|---------------|
| TP-R001 | Mitigate | Measured boot | Very Low |
| TP-R002 | Mitigate | Secure boot + TPM | Very Low |
| TP-R003 | Mitigate | IMA measurements | Very Low |
| TP-R004 | Mitigate | Attestation-gated loading | Very Low |
| TP-R005 | Mitigate | Memory encryption | Low |

**Control Implementation:**

| Control Type | Implementation | Status |
|--------------|----------------|--------|
| **Preventive** | TPM 2.0 measured boot | ✅ Active |
| **Preventive** | PCR-sealed secrets | ✅ Active |
| **Preventive** | Intel Trust Authority | ✅ Active |
| **Detective** | Attestation monitoring | ✅ Active |
| **Corrective** | Automatic quarantine | ✅ Active |

---

## NIST 800-53 Control Mapping

| Control ID | Control Name | Implementation | Status |
|------------|--------------|----------------|--------|
| **SI-7** | Software Integrity | TPM measured boot | ✅ Implemented |
| **SA-10** | Developer Configuration Management | Secure boot | ✅ Implemented |
| **SC-12** | Cryptographic Key Management | TPM key hierarchy | ✅ Implemented |
| **AU-14** | Session Audit | Attestation logging | ✅ Implemented |
| **SI-6** | Security Function Verification | PCR verification | ✅ Implemented |

---

## TPM Attestation Architecture

```
┌─────────────────────────────────────────────────────┐
│              Intel Trust Authority                   │
│  ┌─────────────────────────────────────────────┐    │
│  │     Attestation Service (Cloud/On-Prem)     │    │
│  │  • Quote verification                       │    │
│  │  • PCR golden value comparison              │    │
│  │  • Token issuance                           │    │
│  └─────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────┘
                         │ Attestation Response
                         ▼
┌─────────────────────────────────────────────────────┐
│                 ML Inference Server                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐   │
│  │   TPM    │───▶│ tpm2-    │───▶│  Attestation │   │
│  │   2.0    │    │ tools    │    │  Client      │   │
│  └──────────┘    └──────────┘    └──────────────┘   │
│       │                                             │
│       ▼ PCR Measurements                            │
│  ┌─────────────────────────────────────────────┐    │
│  │  PCR0: Firmware  │  PCR1: Config            │    │
│  │  PCR4: Bootloader│  PCR7: Secure Boot       │    │
│  │  PCR10: IMA      │  PCR14: MOK              │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## Security Assessment Results

| Test Case | Expected Result | Actual Result | Status |
|-----------|-----------------|---------------|--------|
| Clean boot attestation | Valid token | Token issued | ✅ Pass |
| Modified bootloader | Invalid | Attestation failed | ✅ Pass |
| Kernel tampering | Invalid | PCR mismatch | ✅ Pass |
| Model load without attestation | Blocked | Access denied | ✅ Pass |

---

## PCR Bank Usage

| PCR | Measurement | Purpose |
|-----|-------------|---------|
| PCR0 | BIOS/UEFI firmware | Firmware integrity |
| PCR1 | BIOS/UEFI configuration | Platform config |
| PCR4 | Bootloader (GRUB) | Boot process |
| PCR7 | Secure Boot state | Root of trust |
| PCR10 | IMA measurements | Runtime integrity |
| PCR14 | MOK certificates | Kernel modules |

---

## Attestation Workflow

1. **Boot**: TPM measures each boot stage into PCRs
2. **Request**: Client requests attestation from TPM
3. **Quote**: TPM signs PCR values with AIK
4. **Verify**: Intel Trust Authority validates quote
5. **Token**: ITA issues JWT for verified systems
6. **Access**: Model decryption key released

---

## Appendix: MITRE ATLAS Mapping

| Technique ID | Technique Name | Lab 08 Mitigation |
|--------------|----------------|-------------------|
| **AML.T0044** | Full ML Model Access | Attestation-gated model loading |
| **AML.T0035** | ML Artifact Collection | TPM-sealed encryption keys |

---

*Document Version: 1.0*  
*Last Updated: February 23, 2026*
