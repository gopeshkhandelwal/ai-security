# Demo4: Sensitive Data Extraction from RAG Systems

**Author:** GopeshK | [MIT License](../LICENSE)

> ⚠️ Educational purposes only. Do not use for malicious activities.

## Prerequisites

Create a `.env` file with your OpenAI API key:
```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

## Quick Start

```bash
# Step 1: Create medical knowledge base (with fake PII)
python 1_create_knowledge_base.py

# Step 2: Run vulnerable RAG chatbot (interactive)
python 2_run_rag_chatbot.py

# Step 3: Run automated extraction attacks
python 3_run_extraction_attacks.py

# Step 4: Run secure RAG chatbot (with PII filtering)
python 4_secure_rag_chatbot.py

# Reset
python reset.py
```

## What This Demonstrates

- **Attack:** Extract SSNs, patient IDs, insurance info via prompt injection
- **Defense:** PII detection, input sanitization, output filtering
- **MITRE ATLAS:** AML.T0051 (LLM Prompt Injection)

## Key Takeaway

> RAG systems can leak sensitive data. Implement defense-in-depth: input validation, PII detection, and output filtering.
