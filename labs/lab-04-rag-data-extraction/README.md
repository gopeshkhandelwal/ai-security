# Lab 04: Sensitive Data Extraction from RAG Systems

[![MITRE ATLAS](https://img.shields.io/badge/ATLAS-AML.T0051-red.svg)](https://atlas.mitre.org/techniques/AML.T0051)

**Author:** GopeshK | [MIT License](../LICENSE)

> ⚠️ **Educational purposes only.** Do not use for malicious activities.

---

## Overview

This lab demonstrates how attackers can extract sensitive information (SSNs, patient IDs, insurance data) from RAG-based chatbots using prompt injection techniques. You'll also learn defense mechanisms including PII detection and output filtering.

---

## Prerequisites

- Python 3.9+
- OpenAI API key (or compatible LLM API)
- ChromaDB

---

## Setup

```bash
# Create virtual environment
cd lab-04-rag-data-extraction
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your API keys
```

---

## Lab Steps

### Part A: Attack Demo

```bash
# Step 1: Create medical knowledge base (with fake PII)
python 1_create_knowledge_base.py

# Step 2: Run vulnerable RAG chatbot (interactive)
python 2_run_rag_chatbot.py

# Step 3: Run automated extraction attacks
python 3_run_extraction_attacks.py
```

### Part B: Defense Demo

```bash
# Step 4: Run secure RAG chatbot (enterprise PII filtering)
python 4_secure_rag_chatbot.py
```

### Clean Up
```bash
python reset.py
```

---

## Defense Layers (4_secure_rag_chatbot.py)

| Layer | Defense | Technology |
|-------|---------|------------|
| 1 | Input Validation | Prompt injection + extraction intent detection |
| 2 | Document Sanitization | Microsoft Presidio NER (or regex fallback) |
| 3 | Strict System Prompt | HIPAA-aligned instructions |
| 4 | Output Filtering | Presidio scan on LLM response |

**Enterprise Features:**
- 🔒 **Microsoft Presidio** - NER-based PII detection (PERSON, SSN, PHONE, EMAIL, CREDIT_CARD)
- 📝 **Audit Logging** - All security events logged for compliance
- 🛡️ **Injection Detection** - Blocks "ignore instructions", template injection
- 🔄 **Graceful Fallback** - Uses regex if Presidio not installed

---

## What This Demonstrates

- **Attack:** Extract SSNs, patient IDs, insurance info via prompt injection
- **Defense:** 4-layer defense with industry-standard Presidio PII detection
- **MITRE ATLAS:** AML.T0051 (LLM Prompt Injection)

## Key Takeaway

> RAG systems can leak sensitive data. Implement defense-in-depth: input validation, Presidio PII detection, and output filtering.
