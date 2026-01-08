# AI Security Demos

**Author:** GopeshK | [MIT License](LICENSE)

> ⚠️ **DISCLAIMER:** This code is for EDUCATIONAL purposes only. Do not use for malicious purposes.

---

## Demos

| Demo | Description | Link |
|------|-------------|------|
| Demo1 | Malicious Code Injection in ML Models | [📖 README](Demo1/README.md) |
| Demo2 | Model Signing & Tampering Detection | [📖 README](Demo2/README.md) |
| Demo3 | Model Stealing via Query Attack | [📖 README](Demo3/README.md) |
| Demo4 | RAG Sensitive Data Extraction | [📖 README](Demo4/README.md) |

---

## MITRE ATLAS Techniques Covered

| Technique | Description | Demo |
|-----------|-------------|------|
| AML.T0010 | ML Supply Chain Compromise | Demo1, Demo2 |
| AML.T0011 | Backdoor ML Model | Demo1, Demo2 |
| AML.T0044 | Full ML Model Access | Demo3 |
| AML.T0024 | Exfiltration via ML Inference API | Demo3 |
| AML.T0051 | LLM Prompt Injection | Demo4 |

---

## Reset All Demos

```bash
python reset_all.py
```
