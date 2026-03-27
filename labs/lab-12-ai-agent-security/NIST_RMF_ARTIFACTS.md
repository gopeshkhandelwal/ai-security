# NIST RMF Artifacts for Lab 12: AI Agent Security

**System Name:** Secure AI Agent Framework  
**System Owner:** AI Platform Security Team  
**Date:** March 27, 2026  
**Classification:** High Impact (FIPS 199)

---

## NIST AI Risk Management Framework (AI RMF)

### GOVERN: AI Governance and Accountability

**AI Governance Policy Statement:**  
All AI agents operating with autonomous capabilities must implement the five security principles: Principle of Least Privilege (PoLP), Human-in-the-Loop (HITL), Policy as Code (PaC), Autonomy Boundaries, and Auditability. No agent may operate without proper identity, policy enforcement, and audit logging.

**Roles and Responsibilities:**

| Role | Responsibility | Accountable Person |
|------|---------------|-------------------|
| AI System Owner | Agent deployment authorization | Director, AI Platform |
| Identity Manager | Agent identity lifecycle | Identity & Access Lead |
| Policy Administrator | Policy definition and updates | Security Policy Lead |
| HITL Reviewer | Approval of high-risk actions | Designated Approvers |
| Audit Officer | Review of agent action logs | Compliance Officer |
| Security Engineer | Implementation of controls | Security Engineering Lead |

**AI Risk Tolerance:**

| Risk Category | Tolerance Level | Rationale |
|---------------|-----------------|-----------|
| Unauthorized Actions | Zero Tolerance | Agent must operate within policy |
| Privilege Escalation | Zero Tolerance | Agent cannot exceed granted permissions |
| Unaudited Operations | Zero Tolerance | All actions must be traceable |
| Unapproved Destructive Actions | Zero Tolerance | High-risk actions require HITL |
| Policy Bypass | Zero Tolerance | Policy engine must be active |

---

### MAP: AI Risk Identification

**System Context:**  
AI agents with autonomous capabilities can perform actions on behalf of users, including file operations, command execution, network requests, and financial transactions. Without proper security controls, these agents pose significant risks including unauthorized data access, system compromise, and financial loss.

**Threat Actors:**

| Actor | Motivation | Capability | Likelihood |
|-------|------------|------------|------------|
| Malicious User | Privilege escalation | Medium | High |
| Prompt Injector | Agent manipulation | Medium | High |
| Insider Threat | Data exfiltration | High | Medium |
| External Attacker | System compromise | High | Medium |
| Rogue Agent | Autonomous harm | Low | Low |

**AI-Specific Risk Catalog:**

| Risk ID | Risk Description | MITRE ATLAS | Impact | Likelihood |
|---------|------------------|-------------|--------|------------|
| AS-R001 | Agent operates without unique identity | - | Critical | High |
| AS-R002 | Agent has excessive permissions | AML.T0048 | Critical | High |
| AS-R003 | Destructive actions without approval | - | Critical | Medium |
| AS-R004 | Policy bypass allows unauthorized actions | AML.T0054 | Critical | Medium |
| AS-R005 | Actions not traceable to user/agent | - | High | Medium |
| AS-R006 | Agent credentials shared across instances | - | High | High |
| AS-R007 | Time-unbounded agent sessions | - | Medium | Medium |
| AS-R008 | Resource scope not enforced | - | High | Medium |

---

### MEASURE: Risk Assessment and Verification

**Risk Measurement Methodology:**

| Measurement Type | Method | Frequency | Tool |
|------------------|--------|-----------|------|
| Identity Verification | Automated testing | Continuous | CI/CD Pipeline |
| Permission Testing | Penetration testing | Weekly | Security Team |
| Policy Compliance | Policy evaluation audit | Daily | Policy Engine |
| HITL Effectiveness | Approval rate analysis | Weekly | Audit System |
| Audit Completeness | Log integrity verification | Daily | SIEM |

**Protection Metrics:**

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Agent identity uniqueness | 100% | 100% | ✅ Met |
| Permission least privilege | 100% | 98.5% | ✅ Met |
| High-risk action approval rate | >95% | 97.2% | ✅ Met |
| Policy decision logging | 100% | 100% | ✅ Met |
| Audit trail completeness | 100% | 100% | ✅ Met |
| Session timeout enforcement | 100% | 99.8% | ✅ Met |

---

### MANAGE: Risk Treatment and Monitoring

**Control Implementation:**

| Control ID | Control Name | NIST SP 800-53 | Implementation |
|------------|--------------|----------------|----------------|
| AS-C001 | Unique Agent Identity | IA-2, IA-4 | Cryptographic identity per agent instance |
| AS-C002 | Minimal Permissions | AC-6 | Role-based permission sets |
| AS-C003 | Resource Scope Limits | AC-3 | ABAC with path restrictions |
| AS-C004 | Human Approval | AC-3(7) | HITL for high-risk actions |
| AS-C005 | Policy Enforcement | AC-3 | OPA-style policy engine |
| AS-C006 | Complete Audit Trail | AU-2, AU-3 | Immutable action logging |
| AS-C007 | Session Management | AC-12 | Time-bounded credentials |
| AS-C008 | Identity Chain | AU-3 | User delegation tracking |

---

## NIST SP 800-53 Control Mapping

### Access Control (AC)

| Control | Title | Implementation |
|---------|-------|----------------|
| AC-2 | Account Management | Agent identity lifecycle management |
| AC-3 | Access Enforcement | Policy engine authorization checks |
| AC-3(7) | Role-Based Access Control | Agent type determines permissions |
| AC-5 | Separation of Duties | Different agent types for different tasks |
| AC-6 | Least Privilege | Minimal permission sets per agent type |
| AC-6(1) | Authorize Access to Security Functions | Admin agents require elevated approval |
| AC-6(9) | Log Use of Privileged Functions | All privileged actions audited |
| AC-12 | Session Termination | Time-bounded agent sessions |

### Identification and Authentication (IA)

| Control | Title | Implementation |
|---------|-------|----------------|
| IA-2 | Identification and Authentication | Unique agent identity per instance |
| IA-4 | Identifier Management | UUID-based agent IDs with signatures |
| IA-5 | Authenticator Management | Cryptographic identity signatures |
| IA-8 | Identification (Non-Organizational Users) | User delegation chain tracking |

### Audit and Accountability (AU)

| Control | Title | Implementation |
|---------|-------|----------------|
| AU-2 | Event Logging | All agent actions logged |
| AU-3 | Content of Audit Records | User ID, Agent ID, action, outcome |
| AU-3(1) | Additional Audit Information | Policy decision, approval status |
| AU-6 | Audit Record Review | Session summary reports |
| AU-9 | Protection of Audit Information | Append-only audit logs |
| AU-12 | Audit Record Generation | Real-time action logging |

---

## MITRE ATLAS Mapping

| Technique ID | Technique Name | Lab Mitigation |
|--------------|----------------|----------------|
| AML.T0048 | Exfiltration via ML | Resource scope limits, audit logging |
| AML.T0051 | Indirect Prompt Injection | Policy enforcement, HITL approval |
| AML.T0054 | LLM Jailbreak | Policy engine, autonomy boundaries |
| AML.T0053 | User Input Manipulation | Identity verification, audit trail |

---

## OWASP LLM Top 10 Mapping

| OWASP ID | Vulnerability | Lab Control |
|----------|---------------|-------------|
| LLM01 | Prompt Injection | Policy enforcement, HITL |
| LLM02 | Insecure Output Handling | Action result validation |
| LLM03 | Training Data Poisoning | N/A (inference only) |
| LLM04 | Model Denial of Service | Session timeouts |
| LLM05 | Supply Chain Vulnerabilities | Identity verification |
| LLM06 | Sensitive Information Disclosure | Resource scope limits |
| LLM07 | Insecure Plugin Design | Secure tool implementation |
| LLM08 | Excessive Agency | Autonomy boundaries, HITL |
| LLM09 | Overreliance | Human approval for critical actions |
| LLM10 | Model Theft | Audit logging |

---

## Compliance Artifacts

### 1. Security Control Assessment Report

**Assessment Date:** March 27, 2026  
**Assessor:** AI Security Team  
**Assessment Method:** Technical testing, code review, penetration testing

| Control | Assessment Result | Evidence |
|---------|-------------------|----------|
| Unique Identity | PASS | Each agent has UUID + signature |
| Least Privilege | PASS | Permission sets per agent type |
| Resource Scope | PASS | ABAC path restrictions enforced |
| HITL Approval | PASS | Destructive actions require approval |
| Policy Enforcement | PASS | All actions evaluated against policy |
| Audit Logging | PASS | JSONL logs with complete context |
| Session Management | PASS | TTL-based session expiration |

### 2. System Security Plan (SSP) Reference

This lab demonstrates controls for:
- **Information System Category:** AI Agent Platform
- **Security Objective:** Confidentiality (High), Integrity (High), Availability (Moderate)
- **Baseline:** NIST SP 800-53 High Baseline

### 3. Continuous Monitoring Strategy

| Monitoring Activity | Frequency | Responsible Party |
|--------------------|-----------|-------------------|
| Agent identity audit | Real-time | Identity System |
| Policy decision review | Daily | Security Team |
| HITL approval analysis | Weekly | Compliance Team |
| Audit log integrity check | Daily | SIEM |
| Permission creep detection | Weekly | IAM Team |
| Session anomaly detection | Real-time | Security Operations |

---

## Lab Exercise Compliance Validation

### Exercise 1: Vulnerable Agent
**Purpose:** Demonstrate risks without security controls  
**Compliance Gap:** AC-2, AC-3, AC-6, AU-2, IA-2 all violated

### Exercise 2: Agent Identity (PoLP)
**Purpose:** Implement unique identities with minimal permissions  
**Controls Addressed:** IA-2, IA-4, AC-6, AC-6(9)

### Exercise 3: Human-in-the-Loop
**Purpose:** Implement approval workflow for high-risk actions  
**Controls Addressed:** AC-3(7), AU-12

### Exercise 4: Policy Engine
**Purpose:** Implement policy-based access control  
**Controls Addressed:** AC-3, AC-5

### Exercise 5: Secure Agent
**Purpose:** Demonstrate all 5 security principles  
**Controls Addressed:** AC-2, AC-3, AC-6, AU-2, AU-3, IA-2, IA-4

---

## References

- [NIST AI Risk Management Framework 1.0](https://www.nist.gov/itl/ai-risk-management-framework)
- [NIST SP 800-53 Rev. 5](https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final)
- [MITRE ATLAS](https://atlas.mitre.org/)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

---

**Document Control:**
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-27 | AI Security Team | Initial release |
