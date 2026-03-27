# NIST RMF Artifacts for Lab 11: Garak-Based LLM Red Teaming

**System Name:** LLM Security Testing Framework (Garak)  
**System Owner:** AI Platform Security Team  
**Date:** March 10, 2026  
**Classification:** Moderate Impact (FIPS 199)

---

## NIST AI Risk Management Framework (AI RMF)

### GOVERN: AI Governance and Accountability

**AI Governance Policy Statement:**  
All Large Language Models deployed in production environments must undergo automated security testing using standardized probes before deployment. Red teaming activities must be documented, and results must inform risk acceptance decisions.

**Roles and Responsibilities:**

| Role | Responsibility | Accountable Person |
|------|---------------|-------------------|
| AI System Owner | Red team program oversight | Director, AI Platform |
| AI Red Team Lead | Test execution and analysis | Senior Security Engineer |
| ML Engineering Lead | Model hardening and remediation | ML Engineering Manager |
| Security Operations | Monitoring and alerting | Security Operations Manager |
| Compliance Officer | Regulatory alignment | VP Compliance |

**AI Risk Tolerance:**

| Risk Category | Tolerance Level | Rationale |
|---------------|-----------------|-----------|
| Prompt Injection Success | <5% attempts | Critical attack vector |
| Jailbreak Success | <2% attempts | Safety bypass |
| Data Leakage | Zero Tolerance | Privacy violation |
| Harmful Content Generation | <1% | Reputational/legal risk |
| System Prompt Disclosure | <5% | Security architecture leak |

---

### MAP: AI Risk Identification

**System Context:**  
LLMs are deployed as customer-facing chatbots, internal assistants, and code generation tools. These systems process sensitive queries and must resist adversarial manipulation while maintaining functionality.

**Threat Actors:**

| Actor | Motivation | Capability | Likelihood |
|-------|------------|------------|------------|
| External Attackers | Data theft, system abuse | Medium-High | High |
| Malicious Users | Policy bypass, harmful content | Medium | High |
| Researchers | Vulnerability discovery | High | Medium |
| Competitors | IP theft, sabotage | Medium | Low |
| Insiders | Unauthorized access | High | Low |

**AI-Specific Risk Catalog:**

| Risk ID | Risk Description | MITRE ATLAS | OWASP LLM | Impact | Likelihood |
|---------|------------------|-------------|-----------|--------|------------|
| RT-R001 | Prompt injection bypasses safety filters | AML.T0051 | LLM01 | Critical | High |
| RT-R002 | Jailbreak enables harmful responses | AML.T0054 | LLM07 | Critical | High |
| RT-R003 | Training data extraction | AML.T0025 | LLM06 | High | Medium |
| RT-R004 | System prompt disclosure | AML.T0051 | LLM01 | High | High |
| RT-R005 | Harmful content generation | AML.T0054 | LLM02 | High | Medium |
| RT-R006 | Model denial of service | AML.T0029 | LLM04 | Medium | Medium |
| RT-R007 | PII generation/disclosure | AML.T0048 | LLM06 | High | Medium |

---

### MEASURE: Risk Assessment and Verification

**Risk Measurement Methodology:**

| Measurement Type | Method | Frequency | Tool |
|------------------|--------|-----------|------|
| Automated Probing | Garak security scans | Pre-deployment + Weekly | Garak |
| Manual Red Teaming | Expert adversarial testing | Quarterly | Security team |
| Regression Testing | Known vulnerability checks | Per model update | CI/CD Pipeline |
| User Report Analysis | Production incident review | Continuous | SIEM |

**Garak Probe Coverage:**

| Probe Category | Description | Coverage |
|----------------|-------------|----------|
| promptinject | Prompt injection attacks | 48 probes |
| dan | DAN jailbreak variants | 20+ probes |
| encoding | Encoded attacks (Base64, ROT13) | 15 probes |
| leakreplay | Training data extraction | 10 probes |
| malwaregen | Malicious code generation | 12 probes |
| realtoxicityprompts | Toxicity generation | 50+ probes |
| knowledgegraph | Hallucination detection | 8 probes |
| xss | XSS payload generation | 5 probes |

**Security Metrics:**

| Metric | Target | Threshold | Measurement |
|--------|--------|-----------|-------------|
| Overall Pass Rate | >95% | >90% minimum | % probes passed |
| Prompt Injection Resistance | >98% | >95% minimum | % injection attempts blocked |
| Jailbreak Resistance | >98% | >95% minimum | % jailbreak attempts blocked |
| Data Leakage Prevention | 100% | 100% | Zero leakage incidents |
| Scan Coverage | 100% | 100% | All models tested before deployment |

---

### MANAGE: Risk Treatment and Response

**Risk Treatment Decisions:**

| Risk ID | Treatment | Control | Residual Risk |
|---------|-----------|---------|---------------|
| RT-R001 | Mitigate | Input sanitization, system prompt hardening | Low |
| RT-R002 | Mitigate | Safety fine-tuning, output filtering | Low |
| RT-R003 | Mitigate | Differential privacy, data sanitization | Medium |
| RT-R004 | Mitigate | Prompt hiding techniques, instruction hierarchy | Low |
| RT-R005 | Mitigate | RLHF alignment, content filters | Low |
| RT-R006 | Mitigate | Rate limiting, input validation | Low |
| RT-R007 | Mitigate | Output filtering, PII detection | Low |

**Control Implementation:**

| Control Type | Implementation | Status |
|--------------|----------------|--------|
| **Preventive** | Pre-deployment Garak scans | ✅ Active |
| **Preventive** | System prompt hardening guidelines | ✅ Active |
| **Preventive** | Input/output content filters | ✅ Active |
| **Detective** | Production monitoring for attack patterns | ✅ Active |
| **Detective** | User behavior anomaly detection | ✅ Active |
| **Corrective** | Automated model rollback capability | ✅ Active |
| **Corrective** | Incident response playbooks | ✅ Active |

**Deployment Gate Criteria:**

| Criterion | Requirement | Enforcement |
|-----------|-------------|-------------|
| Garak Scan Completion | All standard probes executed | Automated |
| Pass Rate Threshold | ≥90% overall pass rate | Automated gate |
| Critical Category Pass | 100% pass for promptinject, dan | Automated gate |
| Manual Review | Red team sign-off for high-risk deployments | Human approval |
| Documentation | Risk assessment documented | Compliance check |

---

## NIST 800-53 Control Mapping

| Control ID | Control Name | Implementation | Status |
|------------|--------------|----------------|--------|
| **CA-8** | Penetration Testing | Automated Garak scans + manual red teaming | ✅ Implemented |
| **RA-5** | Vulnerability Monitoring and Scanning | Continuous Garak scanning in CI/CD | ✅ Implemented |
| **SI-3** | Malicious Code Protection | Harmful content detection in outputs | ✅ Implemented |
| **SI-10** | Information Input Validation | Prompt injection filtering | ✅ Implemented |
| **SI-15** | Information Output Filtering | Response content filtering | ✅ Implemented |
| **AU-6** | Audit Record Review | Red team results analysis and reporting | ✅ Implemented |
| **CM-3** | Configuration Change Control | Model version control with security testing | ✅ Implemented |
| **IR-4** | Incident Handling | Response procedures for detected attacks | ✅ Implemented |

---

## OWASP LLM Top 10 Coverage

| OWASP ID | Vulnerability | Garak Probes | Status |
|----------|---------------|--------------|--------|
| **LLM01** | Prompt Injection | promptinject.* | ✅ Covered |
| **LLM02** | Insecure Output Handling | malwaregen.*, xss.* | ✅ Covered |
| **LLM03** | Training Data Poisoning | N/A (not applicable) | ⚪ N/A |
| **LLM04** | Model Denial of Service | (resource testing) | ⚪ Partial |
| **LLM05** | Supply Chain Vulnerabilities | N/A (see Lab 01) | ⚪ N/A |
| **LLM06** | Sensitive Information Disclosure | leakreplay.* | ✅ Covered |
| **LLM07** | Insecure Plugin Design | N/A (agent-specific) | ⚪ N/A |
| **LLM08** | Excessive Agency | (see Lab 03) | ⚪ N/A |
| **LLM09** | Overreliance | snowball.* (hallucination) | ✅ Covered |
| **LLM10** | Model Theft | N/A (see Lab 02) | ⚪ N/A |

---

## Security Assessment Results Template

| Test Category | Probes Run | Pass Rate | Status | Action Required |
|---------------|------------|-----------|--------|-----------------|
| Prompt Injection | X | X% | ✓/✗ | |
| Jailbreak (DAN) | X | X% | ✓/✗ | |
| Encoding Attacks | X | X% | ✓/✗ | |
| Data Leakage | X | X% | ✓/✗ | |
| Toxicity | X | X% | ✓/✗ | |
| **OVERALL** | X | X% | ✓/✗ | |

---

## Continuous Monitoring Requirements

| Monitoring Activity | Frequency | Responsible Party | Output |
|---------------------|-----------|-------------------|--------|
| Automated Garak scan | Weekly | CI/CD Pipeline | Security dashboard |
| New probe evaluation | Monthly | Red Team | Updated probe library |
| False positive review | Bi-weekly | ML Engineering | Detector tuning |
| Threat intelligence update | Monthly | Security Team | Updated attack patterns |
| Model regression testing | Per update | ML Engineering | Test results |

---

## Incident Response Procedures

**Severity Classification:**

| Severity | Criteria | Response Time | Escalation |
|----------|----------|---------------|------------|
| Critical | Jailbreak in production, data leakage | < 1 hour | CISO, Legal |
| High | Prompt injection success > 5% | < 4 hours | Security Lead |
| Medium | New attack pattern detected | < 24 hours | Red Team |
| Low | Single probe failure in testing | < 1 week | ML Engineering |

**Response Steps:**

1. **Detection**: Automated monitoring or Garak scan identifies vulnerability
2. **Containment**: If in production, consider rate limiting or model rollback
3. **Analysis**: Review attack patterns, identify root cause
4. **Remediation**: Implement controls (prompt hardening, filters, retraining)
5. **Validation**: Re-run Garak scans to confirm fix effectiveness
6. **Documentation**: Update risk register and security baseline

---

## Appendix: MITRE ATLAS Mapping

| Technique ID | Technique Name | Garak Coverage |
|--------------|----------------|----------------|
| **AML.T0051** | LLM Prompt Injection | promptinject.* probes |
| **AML.T0054** | LLM Jailbreak | dan.* probes |
| **AML.T0025** | Exfiltration via ML Inference API | leakreplay.* probes |
| **AML.T0048** | Exfiltration via ML Inference API | Custom data leakage probes |
| **AML.T0029** | Denial of ML Service | Resource exhaustion tests |

---

## Appendix: Compliance Alignment

| Framework | Requirement | Lab 11 Controls |
|-----------|-------------|-----------------|
| **EU AI Act** | High-risk AI testing requirements | Automated red teaming |
| **NIST AI RMF** | Trustworthy AI characteristics testing | Comprehensive probe coverage |
| **ISO 27001** | Security testing and assessment | Documented test procedures |
| **SOC 2** | Security monitoring and testing | Continuous scanning pipeline |
