# NIST RMF Artifacts for Lab 06: Model Signing Defense

**System Name:** Cryptographic Model Signing Infrastructure  
**System Owner:** AI Platform Security Team  
**Date:** February 23, 2026  
**Classification:** High Impact (FIPS 199)

---

## NIST AI Risk Management Framework (AI RMF)

### GOVERN: AI Governance and Accountability

**AI Governance Policy Statement:**  
All production ML models must be cryptographically signed before deployment. Only models with valid signatures from authorized signing keys may be loaded into production systems. Unsigned or tampered models must be automatically rejected.

**Roles and Responsibilities:**

| Role | Responsibility | Accountable Person |
|------|---------------|-------------------|
| AI System Owner | Model signing policy | Director, AI Platform |
| PKI Administrator | Key management | Security Infrastructure Lead |
| ML Engineer | Model signing workflow | ML Engineering Lead |
| Release Manager | Signature verification | DevOps Lead |

**AI Risk Tolerance:**

| Risk Category | Tolerance Level | Rationale |
|---------------|-----------------|-----------|
| Model Tampering | Zero Tolerance | Integrity violation |
| Unsigned Model Load | Zero Tolerance | Provenance required |
| Key Compromise | Very Low Tolerance | Trust anchor |
| Supply Chain Attack | Zero Tolerance | Malware injection |

---

### MAP: AI Risk Identification

**System Context:**  
Model files can be tampered with after creation. Cryptographic signatures provide integrity verification and provenance attestation. Using ECDSA or Sigstore/Cosign ensures models cannot be modified without detection.

**Threat Actors:**

| Actor | Motivation | Capability | Likelihood |
|-------|------------|------------|------------|
| APT | Model backdooring | Very High | Medium |
| Insider | Sabotage | High | Medium |
| Supply Chain | Mass compromise | High | Medium |
| MITM Attacker | Transit tampering | Medium | Low |

**AI-Specific Risk Catalog:**

| Risk ID | Risk Description | MITRE ATLAS | Impact | Likelihood |
|---------|------------------|-------------|--------|------------|
| MS-R001 | Model weight tampering | AML.T0017 | Critical | Medium |
| MS-R002 | Architecture modification | AML.T0017 | Critical | Low |
| MS-R003 | Backdoor injection | AML.T0018 | Critical | Medium |
| MS-R004 | Model substitution attack | AML.T0010 | Critical | Medium |
| MS-R005 | Signature bypass | AML.T0017 | Critical | Low |

---

### MEASURE: Risk Assessment and Verification

**Risk Measurement Methodology:**

| Measurement Type | Method | Frequency | Tool |
|------------------|--------|-----------|------|
| Signature Verification | Automated check | Every load | Cosign/ECDSA |
| Key Rotation | PKI audit | Quarterly | Security team |
| Tampering Detection | Red team | Monthly | Security research |
| Signature Coverage | Inventory audit | Monthly | Compliance |

**Protection Metrics:**

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Signed model coverage | 100% | 100% | ✅ Met |
| Verification success rate | 100% | 100% | ✅ Met |
| Tamper detection rate | 100% | 100% | ✅ Met |
| Key rotation compliance | 100% | 100% | ✅ Met |

---

### MANAGE: Risk Treatment and Response

**Risk Treatment Decisions:**

| Risk ID | Treatment | Control | Residual Risk |
|---------|-----------|---------|---------------|
| MS-R001 | Mitigate | ECDSA signature | Very Low |
| MS-R002 | Mitigate | Full model hash | Very Low |
| MS-R003 | Mitigate | Signature verification | Very Low |
| MS-R004 | Mitigate | Hash comparison | Very Low |
| MS-R005 | Mitigate | Mandatory verification | Very Low |

**Control Implementation:**

| Control Type | Implementation | Status |
|--------------|----------------|--------|
| **Preventive** | Cosign/Sigstore signing | ✅ Active |
| **Preventive** | ECDSA with SHA-256 | ✅ Active |
| **Preventive** | Mandatory verification | ✅ Active |
| **Detective** | Hash comparison alerts | ✅ Active |
| **Corrective** | Reject unsigned models | ✅ Active |

---

## NIST 800-53 Control Mapping

| Control ID | Control Name | Implementation | Status |
|------------|--------------|----------------|--------|
| **SI-7** | Software Integrity | Cryptographic signatures | ✅ Implemented |
| **SC-12** | Cryptographic Key Management | HSM key storage | ✅ Implemented |
| **SC-13** | Cryptographic Protection | ECDSA P-256 | ✅ Implemented |
| **AU-10** | Non-repudiation | Signature logging | ✅ Implemented |
| **SA-12** | Supply Chain Protection | Provenance verification | ✅ Implemented |

---

## Signing Infrastructure

| Component | Technology | Purpose |
|-----------|------------|---------|
| Signing Algorithm | ECDSA P-256 | Digital signature |
| Hash Function | SHA-256 | Content digest |
| Key Storage | HSM / Cosign | Private key protection |
| Certificate | X.509 / Fulcio | Identity binding |
| Transparency | Rekor log | Audit trail |

---

## Security Assessment Results

| Test Case | Expected Result | Actual Result | Status |
|-----------|-----------------|---------------|--------|
| Signed model verification | Valid | Signature valid | ✅ Pass |
| Single byte modification | Invalid | Signature invalid | ✅ Pass |
| Unsigned model load | Rejected | Load blocked | ✅ Pass |
| Wrong key verification | Invalid | Signature invalid | ✅ Pass |

---

## Signing Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Train Model │───▶│ Hash Model  │───▶│ Sign Hash   │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Load Model  │◀───│ Verify Sig  │◀───│ Distribute  │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## Appendix: MITRE ATLAS Mapping

| Technique ID | Technique Name | Lab 06 Mitigation |
|--------------|----------------|-------------------|
| **AML.T0017** | Develop Adversarial ML Attack | Cryptographic integrity |
| **AML.T0018** | Backdoor ML Model | Signature-verified loading |
| **AML.T0010** | ML Supply Chain Compromise | Provenance attestation |

---

*Document Version: 1.0*  
*Last Updated: February 23, 2026*
