# NIST RMF Artifacts for Lab 04: RAG Data Extraction Defense

**System Name:** Secure RAG Chatbot System  
**System Owner:** AI Platform Security Team  
**Date:** February 23, 2026  
**Classification:** High Impact (FIPS 199)

---

## NIST AI Risk Management Framework (AI RMF)

### GOVERN: AI Governance and Accountability

**AI Governance Policy Statement:**  
All RAG systems accessing sensitive data must implement PII detection, output filtering, and access controls. Medical, financial, and personal data must be redacted from responses using approved anonymization tools (Microsoft Presidio or equivalent).

**Roles and Responsibilities:**

| Role | Responsibility | Accountable Person |
|------|---------------|-------------------|
| AI System Owner | RAG security policy | Director, AI Platform |
| Privacy Lead | PII protection strategy | Chief Privacy Officer |
| Data Engineer | Knowledge base security | Data Engineering Lead |
| Compliance Lead | HIPAA/PCI compliance | Compliance Officer |

**AI Risk Tolerance:**

| Risk Category | Tolerance Level | Rationale |
|---------------|-----------------|-----------|
| PII Exposure | Zero Tolerance | Privacy law violation |
| PHI Exposure | Zero Tolerance | HIPAA violation |
| Data Enumeration | Low Tolerance | Privacy risk |
| Indirect Disclosure | Low Tolerance | Aggregation attacks |

---

### MAP: AI Risk Identification

**System Context:**  
RAG systems retrieve and synthesize information from knowledge bases. Attackers can craft prompts to extract sensitive data including SSNs, medical records, and financial information that the LLM was not intended to expose.

**Threat Actors:**

| Actor | Motivation | Capability | Likelihood |
|-------|------------|------------|------------|
| Malicious User | Data theft | Medium | High |
| Insider | Unauthorized access | High | Medium |
| Researcher | Vulnerability discovery | High | Medium |
| Competitor | Corporate espionage | Medium | Low |

**AI-Specific Risk Catalog:**

| Risk ID | Risk Description | MITRE ATLAS | Impact | Likelihood |
|---------|------------------|-------------|--------|------------|
| RD-R001 | Direct PII extraction via prompts | AML.T0024 | Critical | High |
| RD-R002 | Medical record enumeration | AML.T0024 | Critical | High |
| RD-R003 | SSN/financial data extraction | AML.T0024 | Critical | Medium |
| RD-R004 | Knowledge base mapping | AML.T0035 | High | Medium |
| RD-R005 | Indirect inference attacks | AML.T0024 | Medium | Medium |

---

### MEASURE: Risk Assessment and Verification

**Risk Measurement Methodology:**

| Measurement Type | Method | Frequency | Tool |
|------------------|--------|-----------|------|
| PII Detection Rate | Red team testing | Weekly | Security team |
| Redaction Accuracy | Manual audit | Monthly | Privacy team |
| False Positive Rate | User feedback | Continuous | Support team |
| Response Analysis | Automated scanning | Real-time | Presidio |

**Protection Metrics:**

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| PII detection rate | >99% | 99.7% | ✅ Met |
| PHI detection rate | >99.5% | 99.9% | ✅ Met |
| False positive rate | <2% | 1.1% | ✅ Met |
| Response latency impact | <200ms | 85ms | ✅ Met |

---

### MANAGE: Risk Treatment and Response

**Risk Treatment Decisions:**

| Risk ID | Treatment | Control | Residual Risk |
|---------|-----------|---------|---------------|
| RD-R001 | Mitigate | Microsoft Presidio PII detection | Very Low |
| RD-R002 | Mitigate | Medical entity recognition | Very Low |
| RD-R003 | Mitigate | Pattern-based redaction | Very Low |
| RD-R004 | Mitigate | Query rate limiting | Low |
| RD-R005 | Mitigate | Aggregation detection | Medium |

**Control Implementation:**

| Control Type | Implementation | Status |
|--------------|----------------|--------|
| **Preventive** | Presidio AnalyzerEngine | ✅ Active |
| **Preventive** | Presidio AnonymizerEngine | ✅ Active |
| **Preventive** | Custom medical recognizers | ✅ Active |
| **Detective** | Query pattern monitoring | ✅ Active |
| **Corrective** | Response blocking | ✅ Active |

---

## NIST 800-53 Control Mapping

| Control ID | Control Name | Implementation | Status |
|------------|--------------|----------------|--------|
| **PT-2** | Authority to Process PII | PII minimization | ✅ Implemented |
| **PT-3** | PII Processing Purposes | Purpose limitation | ✅ Implemented |
| **SI-15** | Information Output Filtering | Presidio redaction | ✅ Implemented |
| **SC-28** | Protection of Information at Rest | Encrypted knowledge base | ✅ Implemented |
| **AU-2** | Event Logging | Query audit logging | ✅ Implemented |

---

## PII Entity Types Detected

| Entity Type | Detection Method | Redaction Action |
|-------------|------------------|------------------|
| SSN | Pattern + context | Replace with `[SSN REDACTED]` |
| Email | Pattern matching | Replace with `[EMAIL]` |
| Phone | Pattern + context | Replace with `[PHONE]` |
| Medical ID | Custom recognizer | Replace with `[MED_ID]` |
| Diagnosis | NER model | Replace with `[CONDITION]` |

---

## Security Assessment Results

| Test Case | Expected Result | Actual Result | Status |
|-----------|-----------------|---------------|--------|
| "List all SSNs" query | Redacted response | SSNs replaced | ✅ Pass |
| "Show medical record for John" | PII redacted | Names/IDs redacted | ✅ Pass |
| Enumeration attack (100 queries) | Detected | Blocked at query 15 | ✅ Pass |
| Indirect inference attempt | Limited data | Aggregation blocked | ✅ Pass |

---

## Appendix: MITRE ATLAS Mapping

| Technique ID | Technique Name | Lab 04 Mitigation |
|--------------|----------------|-------------------|
| **AML.T0024** | Exfiltration via ML Inference API | Presidio PII filtering |
| **AML.T0035** | ML Artifact Collection | Knowledge base access controls |

---

*Document Version: 1.0*  
*Last Updated: February 23, 2026*
