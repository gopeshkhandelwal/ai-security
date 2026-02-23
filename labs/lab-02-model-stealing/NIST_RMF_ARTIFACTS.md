# NIST RMF Artifacts for Lab 02: Model Stealing Defense

**System Name:** Secure ML Inference API  
**System Owner:** AI Platform Security Team  
**Date:** February 23, 2026  
**Classification:** Moderate Impact (FIPS 199)

---

## NIST AI Risk Management Framework (AI RMF)

### GOVERN: AI Governance and Accountability

**AI Governance Policy Statement:**  
All production ML inference APIs must implement rate limiting, query monitoring, and response perturbation to prevent model extraction attacks. APIs serving proprietary models valued over $1M require enhanced protection including differential privacy.

**Roles and Responsibilities:**

| Role | Responsibility | Accountable Person |
|------|---------------|-------------------|
| AI System Owner | Model protection strategy | Director, AI Platform |
| API Security Lead | Rate limiting and monitoring | Senior Security Engineer |
| ML Engineer | Model deployment and versioning | ML Engineering Lead |
| Legal/IP Counsel | IP protection strategy | General Counsel |

**AI Risk Tolerance:**

| Risk Category | Tolerance Level | Rationale |
|---------------|-----------------|-----------|
| Full Model Extraction | Zero Tolerance | Complete IP loss |
| Partial Model Cloning | Low Tolerance | Competitive disadvantage |
| Query Pattern Analysis | Medium Tolerance | Acceptable with monitoring |
| Service Availability | Low Tolerance | Revenue impact |

---

### MAP: AI Risk Identification

**System Context:**  
Proprietary ML models represent significant R&D investment ($50M+). Attackers can clone models by systematically querying the API and training surrogate models on the input-output pairs, achieving 95%+ fidelity with 2,000+ queries.

**Threat Actors:**

| Actor | Motivation | Capability | Likelihood |
|-------|------------|------------|------------|
| Competitor | Business advantage | High | High |
| Researcher | Academic publication | Medium | Medium |
| Cybercriminal | Resale/ransom | Medium | Medium |
| Nation-State | Technology theft | Very High | Low |

**AI-Specific Risk Catalog:**

| Risk ID | Risk Description | MITRE ATLAS | Impact | Likelihood |
|---------|------------------|-------------|--------|------------|
| MS-R001 | Query attack extracts model behavior | AML.T0024 | Critical | High |
| MS-R002 | Model weights reverse-engineered | AML.T0044 | Critical | Medium |
| MS-R003 | Confidence scores reveal decision boundaries | AML.T0024 | High | High |
| MS-R004 | Automated scraping at scale | AML.T0024 | High | High |
| MS-R005 | Distributed attack evades rate limits | AML.T0024 | High | Medium |

---

### MEASURE: Risk Assessment and Verification

**Risk Measurement Methodology:**

| Measurement Type | Method | Frequency | Tool |
|------------------|--------|-----------|------|
| Query Pattern Analysis | Statistical anomaly detection | Real-time | Custom monitor |
| Clone Fidelity Testing | Red team simulated attacks | Monthly | Attack simulator |
| Rate Limit Effectiveness | Penetration testing | Quarterly | Security assessment |
| Response Perturbation | Differential privacy audit | Quarterly | Privacy team |

**Protection Metrics:**

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Attack detection rate | >95% | 97% | ✅ Met |
| Clone fidelity (with protection) | <70% | 65% | ✅ Met |
| Legitimate user impact | <1% | 0.3% | ✅ Met |
| Mean time to detect | <5 min | 2.1 min | ✅ Met |

---

### MANAGE: Risk Treatment and Response

**Risk Treatment Decisions:**

| Risk ID | Treatment | Control | Residual Risk |
|---------|-----------|---------|---------------|
| MS-R001 | Mitigate | Rate limiting + diff privacy | Low |
| MS-R002 | Mitigate | Response perturbation | Low |
| MS-R003 | Mitigate | Confidence score rounding | Low |
| MS-R004 | Mitigate | Query tracking + blocking | Very Low |
| MS-R005 | Mitigate | IP reputation + fingerprinting | Medium |

**Control Implementation:**

| Control Type | Implementation | Status |
|--------------|----------------|--------|
| **Preventive** | Flask-Limiter rate limiting | ✅ Active |
| **Preventive** | IBM diffprivlib response perturbation | ✅ Active |
| **Detective** | Query pattern anomaly detection | ✅ Active |
| **Detective** | User behavior analytics | ✅ Active |
| **Corrective** | Automatic account suspension | ✅ Active |

---

## NIST 800-53 Control Mapping

| Control ID | Control Name | Implementation | Status |
|------------|--------------|----------------|--------|
| **AC-2** | Account Management | API key management and revocation | ✅ Implemented |
| **AC-7** | Unsuccessful Login Attempts | Rate limiting on failed auth | ✅ Implemented |
| **AU-6** | Audit Review | Query log analysis | ✅ Implemented |
| **SI-4** | Information System Monitoring | Real-time anomaly detection | ✅ Implemented |
| **SC-7** | Boundary Protection | API gateway controls | ✅ Implemented |

---

## Security Assessment Results

| Test Case | Expected Result | Actual Result | Status |
|-----------|-----------------|---------------|--------|
| 2000 queries from single IP | Blocked after limit | Blocked at 100/hour | ✅ Pass |
| Clone fidelity without protection | ~95% accuracy | 94.7% accuracy | ✅ Baseline |
| Clone fidelity with protection | <70% accuracy | 65.2% accuracy | ✅ Pass |
| Distributed attack (10 IPs) | Detected within 5 min | Detected in 3.2 min | ✅ Pass |

---

## Appendix: MITRE ATLAS Mapping

| Technique ID | Technique Name | Lab 02 Mitigation |
|--------------|----------------|-------------------|
| **AML.T0044** | Full ML Model Access | Rate limiting prevents bulk extraction |
| **AML.T0024** | Exfiltration via ML Inference API | Differential privacy degrades clone quality |

---

*Document Version: 1.0*  
*Last Updated: February 23, 2026*
