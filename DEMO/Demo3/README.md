# Demo 5: Model Stealing via Query Attack

## MITRE ATLAS ATT&CK Techniques Demonstrated

- **AML.T0044** - Full ML Model Access (attempted via query attack)
- **AML.T0024** - Exfiltration via ML Inference API
- **AML.T0035** - ML Model Inference API Access

## Attack Scenario

An attacker with query access to a proprietary ML model (API access) can **steal the model's intellectual property** by:
1. Sending many queries to the victim model via HTTP API
2. Collecting input-output pairs from API responses
3. Training a surrogate/clone model that mimics the victim

This is a **model extraction attack** - the attacker doesn't need the original training data or model architecture!

## Demo Flow (< 1 minute on CPU)

### Option A: All-in-One Demo (Simulated API)
```bash
python run_demo.py
```

### Option B: Real HTTP API Demo (Two Terminals)

**Terminal 1 - Start the API Server:**
```bash
python 1_victim_model.py      # Create the model
python 1b_api_server.py       # Start HTTP API on port 5000
```

**Terminal 2 - Run the Attack:**
```bash
python 2_query_attack.py      # Attack via HTTP requests
python 3_compare_models.py    # Analyze the theft
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single loan application |
| `/batch_predict` | POST | Batch predictions |
| `/api_info` | GET | API documentation |
| `/health` | GET | Health check |

## Key Takeaway

> "Query access to ML APIs is enough to steal them. Implement rate limiting, query auditing, and differential privacy to protect your ML intellectual property."

## Reset Demo
```bash
python reset.py
```
