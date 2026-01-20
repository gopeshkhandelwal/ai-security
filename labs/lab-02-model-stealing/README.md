# Lab 02: Model Stealing via Query Attack

[![MITRE ATLAS](https://img.shields.io/badge/ATLAS-AML.T0044-red.svg)](https://atlas.mitre.org/techniques/AML.T0044)
[![MITRE ATLAS](https://img.shields.io/badge/ATLAS-AML.T0024-red.svg)](https://atlas.mitre.org/techniques/AML.T0024)

**Author:** GopeshK | [MIT License](../LICENSE)

> ⚠️ **Educational purposes only.** Do not use for malicious activities.

---

## Overview

This lab demonstrates how attackers can clone ML models by querying their APIs. With just 2000 queries, an attacker can create a surrogate model with ~95% fidelity.

---

## Prerequisites

- Python 3.9+
- Flask
- scikit-learn

---

## Setup

```bash
# Create virtual environment
cd lab-02-model-stealing
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Lab Steps

### Part A: Attack Demo

#### Terminal 1: Start Vulnerable API Server
```bash
python 1_proprority_model.py    # Create proprietary loan model
python 1b_api_server.py         # Start API on port 5000 (keep running)
```

#### Terminal 2: Run Attack
```bash
python 2_query_attack.py        # Steal model via HTTP queries
python 3_compare_models.py      # Compare stolen vs original (~95% fidelity)
```

### Part B: Defense Demo

#### Terminal 1: Start Secure API Server
```bash
# Stop previous server (Ctrl+C), then:
python 4_secure_api_server.py   # Start secure API on port 5000
```

#### Terminal 2: Run Same Attack
```bash
python 2_query_attack.py        # Attack degraded (~65% fidelity)
```

### Clean Up
```bash
python reset.py
```

---

## Defense Layers (4_secure_api_server.py)

| Layer | Defense | Technology |
|-------|---------|------------|
| 1 | Rate Limiting | Flask-Limiter (blocks after 100 req/min) |
| 2 | Query Tracking | Flags IPs with >20 req/min as suspicious |
| 3 | Differential Privacy | IBM diffprivlib (adds noise to suspicious requests) |
| 4 | Audit Logging | All suspicious activity logged |

**Behavior:**
| Requests/min | Status | Response |
|--------------|--------|----------|
| < 20 | ✅ Normal | Clean prediction |
| 20-100 | ⚠️ Suspicious | Noisy prediction (DP applied) |
| > 100 | ⛔ Blocked | 429 Rate Limited |

---

## What This Demonstrates

- **Attack:** Clone a model by querying its API 2000 times
- **Result (Vulnerable):** Attacker gets surrogate model with ~95% fidelity
- **Result (Secure):** Attack degraded to ~65% fidelity + logged
- **MITRE ATLAS:** AML.T0044 (Full Model Access), AML.T0024 (Exfiltration via API)

## Key Takeaway

> Query access to ML APIs is enough to steal them. Implement rate limiting, query auditing, and differential privacy.
