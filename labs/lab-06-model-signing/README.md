# Lab 06: Model Signing & Tampering Detection

[![MITRE ATLAS](https://img.shields.io/badge/ATLAS-AML.T0010-red.svg)](https://atlas.mitre.org/techniques/AML.T0010)
[![MITRE ATLAS](https://img.shields.io/badge/ATLAS-AML.T0011-red.svg)](https://atlas.mitre.org/techniques/AML.T0011)

**Author:** GopeshK | [MIT License](../LICENSE)

> ⚠️ **Educational purposes only.** Do not use for malicious activities.

---

## Overview

This lab demonstrates how cryptographic signatures can protect ML models from tampering. You'll learn to sign models with ECDSA and detect unauthorized modifications.

---

## Prerequisites

- Python 3.9+
- cryptography library

---

## Setup

```bash
# Create virtual environment
cd lab-06-model-signing
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Lab Steps

```bash
# Step 1: Train a benign model
python 1_train_model.py

# Step 2: Sign the model (creates signature)
python 2_sign_model.py

# Step 3: Attacker tampers with model
python 3_tamper_model.py

# Step 4: Demo without verification (shows attack impact)
python 4_consume_model.py --unverified

# Step 5: Verify again (FAILS - tampering detected!)
python 4_consume_model.py



# Reset
python reset.py
```

## What This Demonstrates

- **Defense:** ECDSA cryptographic signatures for ML models
- **Attack Blocked:** Model tampering detected via signature verification
- **MITRE ATLAS:** AML.T0010 (Supply Chain), AML.T0011 (Backdoor)

## Key Takeaway

> Cryptographic signing of ML models enables consumers to detect supply chain attacks.
