# AI Security

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](labs/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![MITRE ATLAS](https://img.shields.io/badge/Framework-MITRE%20ATLAS-red.svg)](https://atlas.mitre.org/)

A comprehensive collection of hands-on labs and resources for learning AI/ML security, aligned with the [MITRE ATLAS](https://atlas.mitre.org/) adversarial threat framework.

---

## 📁 Repository Structure

```
ai-security/
├── labs/                          # Hands-on security labs
│   ├── lab-01-malicious-code-injection/
│   ├── lab-02-model-signing/
│   ├── lab-03-model-stealing/
│   ├── lab-04-rag-data-extraction/
│   ├── lab-05-llm-agent-exploitation/
│   └── lab-06-supply-chain-attack/
└── README.md                      # This file
```

---

## 🧪 Labs Overview

| Lab | Topic | MITRE ATLAS Techniques |
|-----|-------|------------------------|
| [Lab 01](labs/lab-01-malicious-code-injection/) | Malicious Code Injection | AML.T0010, AML.T0011 |
| [Lab 02](labs/lab-02-model-signing/) | Model Signing & Integrity | AML.T0010, AML.T0011 |
| [Lab 03](labs/lab-03-model-stealing/) | Model Stealing via API | AML.T0044, AML.T0024 |
| [Lab 04](labs/lab-04-rag-data-extraction/) | RAG Data Extraction | AML.T0051 |
| [Lab 05](labs/lab-05-llm-agent-exploitation/) | LLM Agent Exploitation | AML.T0051, AML.T0043 |
| [Lab 06](labs/lab-06-supply-chain-attack/) | HuggingFace Supply Chain Attack | AML.T0010, AML.T0011 |

---

## 🚀 Quick Start

```bash
# Clone repository
git clone <repo-url>
cd ai-security/labs

# Start with Lab 01
cd lab-01-malicious-code-injection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## ⚠️ Disclaimer

> **This repository is for EDUCATIONAL and RESEARCH purposes only.**
> 
> Do not use any code, techniques, or materials for malicious activities.
> The author assumes no liability for misuse.

---

## 📄 License

[MIT License](labs/LICENSE)

---

## 👤 Author

**GopeshK**
