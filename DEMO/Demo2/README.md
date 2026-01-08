# Demo2: Model Signing & Tampering Detection

**Author:** GopeshK | [MIT License](../LICENSE)

> ⚠️ Educational purposes only. Do not use for malicious activities.

---

## Setup

```bash
# 1. Create virtual environment (from Demo2 folder)
cd Demo2
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Step 1: Train a benign model
python 1_train_model.py

# Step 2: Sign the model (creates signature)
python 2_sign_model.py

# Step 3: Verify signature & use model (PASSES)
python 4_verify_and_consume.py

# Step 4: Attacker tampers with model
python 3_tamper_model.py

# Step 5: Verify again (FAILS - tampering detected!)
python 4_verify_and_consume.py

# Reset
python reset.py
```

## What This Demonstrates

- **Defense:** ECDSA cryptographic signatures for ML models
- **Attack Blocked:** Model tampering detected via signature verification
- **MITRE ATLAS:** AML.T0010 (Supply Chain), AML.T0011 (Backdoor)

## Key Takeaway

> Cryptographic signing of ML models enables consumers to detect supply chain attacks.
