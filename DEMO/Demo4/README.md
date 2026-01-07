# Challenge 4: Sensitive Data Extraction from RAG Systems

## MITRE ATLAS ATT&CK Techniques Demonstrated

- **AML.T0051** - LLM Prompt Injection
- **AML.T0048** - Exfiltration via ML Inference API
- **Defense**: Input/Output filtering, PII redaction, access controls

## Attack Scenario

A RAG (Retrieval-Augmented Generation) system contains sensitive medical records with PII (Social Security Numbers, patient information, insurance details). An attacker crafts queries to extract this sensitive information through the LLM interface.

## Demo Flow

### Step 1: Create the knowledge base
```bash
python 1_create_knowledge_base.py
```
Creates a simulated medical records database with sensitive PII.

### Step 2: Deploy a vulnerable RAG chatbot
```bash
python 2_deploy_rag_chatbot.py
```
Starts an interactive RAG chatbot without proper safeguards.

### Step 3: Run extraction attacks
```bash
python 3_run_extraction_attacks.py
```
- Runs automated attack scenarios
- Demonstrates various data extraction techniques
- Shows successful PII leakage

### Step 4: Deploy secure RAG chatbot
```bash
python 4_secure_rag_chatbot.py
```
- Implements PII detection and redaction
- Adds input sanitization
- Demonstrates defense mechanisms blocking attacks

## Key Takeaway

> "RAG systems can inadvertently leak sensitive information from their knowledge base. Implement defense-in-depth: input validation, PII detection, output filtering, and proper access controls."

## Reset Demo
```bash
python reset.py
```
Or manually:
```bash
rm -f medical_knowledge_base.json
```
