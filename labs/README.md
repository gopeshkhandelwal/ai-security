# AI Security Labs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![MITRE ATLAS](https://img.shields.io/badge/MITRE-ATLAS-red.svg)](https://atlas.mitre.org/)

**Author:** GopeshK

> ⚠️ **DISCLAIMER:** This repository is for **EDUCATIONAL** and **RESEARCH** purposes only. Do not use for malicious activities.

---

## Overview

Hands-on security labs demonstrating common AI/ML attack vectors and defense mechanisms, aligned with the [MITRE ATLAS](https://atlas.mitre.org/) framework.

---

## Labs

| Lab | Topic | Attack | Defense | Link |
|-----|-------|--------|---------|------|
| 01 | Supply Chain | HuggingFace trust_remote_code | 5-Layer Security Scanner | [📖 README](lab-01-supply-chain-attack/README.md) |
| 02 | Exfiltration | Model Stealing (Query) | Rate Limiting + Differential Privacy | [📖 README](lab-02-model-stealing/README.md) |
| 03 | LLM Agents | Indirect Prompt Injection | LLM-as-a-Judge Guardrail | [📖 README](lab-03-llm-agent-exploitation/README.md) |
| 04 | Data Leakage | RAG PII Extraction | Microsoft Presidio + 4-Layer Defense | [📖 README](lab-04-rag-data-extraction/README.md) |
| 05 | Supply Chain | Malicious Lambda Injection | ModelScan + Safe Loading | [📖 README](lab-05-malicious-code-injection/README.md) |
| 06 | Integrity | Model Tampering | ECDSA Signing | [📖 README](lab-06-model-signing/README.md) |

---

## Industry Security Tools Used

| Tool | Vendor | Used In |
|------|--------|---------|
|------|--------|---------|n| ModelScan | Protect AI | Lab 05 |
| Flask-Limiter | Pallets | Lab 02 |
| diffprivlib | IBM | Lab 02 |
| Presidio | Microsoft | Lab 04 |
| Custom AST Scanner | - | Lab 01 |

---

## MITRE ATLAS Techniques Covered

| Technique | Name | Lab |
|-----------|------|-----|
| [AML.T0010](https://atlas.mitre.org/techniques/AML.T0010) | ML Supply Chain Compromise | Lab 01, Lab 05 |
| [AML.T0011](https://atlas.mitre.org/techniques/AML.T0011) | Backdoor ML Model | Lab 01, Lab 05 |
| [AML.T0044](https://atlas.mitre.org/techniques/AML.T0044) | Full ML Model Access | Lab 02 |
| [AML.T0024](https://atlas.mitre.org/techniques/AML.T0024) | Exfiltration via ML Inference API | Lab 02 |
| [AML.T0051](https://atlas.mitre.org/techniques/AML.T0051) | LLM Prompt Injection | Lab 03, Lab 04 |

---

## Quick Start

```bash
# Clone and navigate
git clone <repo-url>
cd ai-security/labs

# Install base dependencies
pip install -r requirements.txt

# Navigate to a lab
cd lab-01-supply-chain-attack
```

---

## Reset All Labs

```bash
python reset_all.py
```

---

## Contributing

Contributions welcome! Please read the lab READMEs and follow the existing code style.

---

## License

[MIT License](LICENSE)
