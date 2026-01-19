# Lab 01: Malicious Code Injection in ML Models

[![MITRE ATLAS](https://img.shields.io/badge/ATLAS-AML.T0010-red.svg)](https://atlas.mitre.org/techniques/AML.T0010)
[![MITRE ATLAS](https://img.shields.io/badge/ATLAS-AML.T0011-red.svg)](https://atlas.mitre.org/techniques/AML.T0011)

**Author:** GopeshK | [MIT License](../LICENSE)

> ⚠️ **Educational purposes only.** Do not use for malicious activities.

---

## Overview

This lab demonstrates how attackers can inject malicious code into ML models via Lambda layers. The injected code executes automatically during model inference, enabling supply chain attacks.

---

## Prerequisites

- Python 3.9+
- TensorFlow/Keras

---

## Setup

```bash
# Create virtual environment
cd lab-01-malicious-code-injection
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add/update keys
```

---

## Lab Steps

### Part A: Attack Demo

```bash
# Step 1: Train a benign model
python 1_train_model.py

# Step 2: Use the model (works normally)
python 3_consume_model.py

# Step 3: Inject malicious code into model
python 2_inject_malicious_code.py

# Step 4: Use the model again (malicious payload executes!)
python 3_consume_model.py
```

### Part B: Defense Demo

```bash
# Step 5: Try secure loading (attack blocked!)
python 4_secure_model_loading.py
```

### Clean Up
```bash
python reset.py
```

---

## Defense Layers (4_secure_model_loading.py)

| Layer | Defense | Technology |
|-------|---------|------------|
| 1 | Security Scan | ModelScan (Protect AI) + Lambda/pattern detection |
| 2 | Hash Verification | SHA256 against known-good registry |
| 3 | Safe Loading | TensorFlow `safe_mode=True` |

---

## What This Demonstrates

- **Attack:** Malicious Lambda layer injected into Keras model
- **Impact:** Code executes automatically during inference
- **Defense:** Pre-load scanning detects and blocks malicious code
- **MITRE ATLAS:** AML.T0010 (Supply Chain), AML.T0011 (Backdoor)

## Key Takeaway

> Always verify model integrity before loading untrusted models. Use model scanning tools like ModelScan (Protect AI).
