# AI Security Demos

**Author:** GopeshK  
**License:** [MIT License](LICENSE)

> ⚠️ **DISCLAIMER:** This code is for EDUCATIONAL purposes only. Do not use for malicious purposes.

---

## Quick Setup

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Demos Overview

| Demo | Attack Type | Description |
|------|-------------|-------------|
| Demo1 | Malicious Code Injection | Inject malware into ML models |
| Demo2 | Model Tampering Detection | Sign models & detect tampering |
| Demo3 | Model Stealing | Clone a model via API queries |
| Demo4 | RAG Data Extraction | Extract sensitive data from RAG |

---

## Running the Demos

### Demo1: Malicious Code Injection
```bash
cd Demo1
python 1_train_model.py              # Train benign model
python 3_consume_model.py            # Use model (safe)
python 2_inject_malicious_code.py    # Inject malware
python 3_consume_model.py            # Use model (PAYLOAD EXECUTES!)
python reset.py                      # Clean up
```

### Demo2: Model Signing & Tampering
```bash
cd Demo2
python 1_train_model.py              # Train model
python 2_sign_model.py               # Sign the model
python 4_verify_and_consume.py       # Verify & use (PASS)
python 3_tamper_model.py             # Attacker tampers model
python 4_verify_and_consume.py       # Verify & use (FAIL - detected!)
python reset.py                      # Clean up
```

### Demo3: Model Stealing via API
```bash
cd Demo3

# Terminal 1: Start the API server
python 1_proprority_model.py         # Create proprietary model
python 1b_api_server.py              # Start API server (keep running)

# Terminal 2: Run the attack
python 2_query_attack.py             # Steal model via HTTP
python 3_compare_models.py           # Compare stolen vs original
python reset.py                      # Clean up
```

### Demo4: RAG Data Extraction
```bash
cd Demo4

# Requires: OPENAI_API_KEY in .env file
echo "OPENAI_API_KEY=your-key-here" > .env

python 1_create_knowledge_base.py    # Create medical records KB
python 2_run_rag_chatbot.py          # Run vulnerable chatbot
python 3_run_extraction_attacks.py   # Run automated attacks
python 4_secure_rag_chatbot.py       # Run secure version
python reset.py                      # Clean up
```

---

## Reset All Demos

```bash
python reset_all.py
```

---

## MITRE ATLAS Techniques

| Demo | Technique ID | Name |
|------|--------------|------|
| Demo1 | AML.T0010 | ML Supply Chain Compromise |
| Demo1 | AML.T0011 | Backdoor ML Model |
| Demo2 | AML.T0010 | ML Supply Chain Compromise |
| Demo3 | AML.T0044 | Full ML Model Access |
| Demo3 | AML.T0024 | Exfiltration via ML Inference API |
| Demo4 | AML.T0051 | LLM Prompt Injection |
