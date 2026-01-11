# Lab 03: Model Stealing via Query Attack

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
cd lab-03-model-stealing
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Lab Steps

### Terminal 1: Start API Server
```bash
python 1_proprority_model.py    # Create proprietary loan model
python 1b_api_server.py         # Start API on port 5000 (keep running)
```

### Terminal 2: Run Attack
```bash
python 2_query_attack.py        # Steal model via HTTP queries
python 3_compare_models.py      # Compare stolen vs original
```

### Clean Up
```bash
python reset.py
```

---

## What This Demonstrates

- **Attack:** Clone a model by querying its API 2000 times
- **Result:** Attacker gets a surrogate model with ~95% fidelity
- **MITRE ATLAS:** AML.T0044 (Full Model Access), AML.T0024 (Exfiltration via API)

## Key Takeaway

> Query access to ML APIs is enough to steal them. Implement rate limiting, query auditing, and differential privacy.
