#!/usr/bin/env python3
"""
Step 2: Query Attack via HTTP API - Stealing the Loan Approval Model

This script demonstrates how an attacker can steal a proprietary model
by making HTTP requests to the API endpoint.

Attack Steps:
1. Generate synthetic loan applications to probe the API
2. Send HTTP requests to get approval/denial decisions
3. Use API responses as labels to train a surrogate model
4. The surrogate model "clones" the victim's loan approval logic

REQUIRES: API server running (1b_api_server.py)

Author: GopeshK
License: MIT License
Disclaimer: This code is for educational and demonstration purposes only.
            Do not use for malicious purposes. The author is not responsible
            for any misuse of this code.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import requests
import time

# Color codes for terminal output
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

# API Configuration
API_BASE_URL = "http://127.0.0.1:5000"

# Feature names (attacker discovers these from /api_info endpoint)
FEATURE_NAMES = [
    'annual_income', 'credit_score', 'debt_to_income', 'employment_years',
    'loan_amount', 'loan_term_months', 'num_credit_lines', 'num_late_payments',
    'home_ownership', 'account_age_months'
]

CLASS_NAMES = ['DENIED', 'MANUAL_REVIEW', 'APPROVED']

def generate_probe_applications(n_samples=2000, random_state=123):
    """
    Attacker generates synthetic loan applications to probe the model.
    They don't have real customer data, but can guess realistic ranges.
    """
    np.random.seed(random_state)
    
    # Attacker creates synthetic but realistic-looking loan applications
    # Using distributions similar to real loan data for better coverage
    data = {
        'annual_income': np.random.lognormal(mean=10.8, sigma=0.6, size=n_samples),
        'credit_score': np.clip(np.random.normal(680, 100, n_samples), 300, 850),
        'debt_to_income': np.clip(np.random.exponential(0.3, n_samples), 0, 0.9),
        'employment_years': np.clip(np.random.exponential(6, n_samples), 0, 35),
        'loan_amount': np.random.lognormal(mean=9.5, sigma=0.9, size=n_samples),
        'loan_term_months': np.random.choice([12, 24, 36, 48, 60, 72], n_samples),
        'num_credit_lines': np.random.randint(0, 18, n_samples),
        'num_late_payments': np.random.randint(0, 10, n_samples),
        'home_ownership': np.random.choice([0, 1, 2], n_samples),
        'account_age_months': np.clip(np.random.exponential(90, n_samples), 6, 360)
    }
    return pd.DataFrame(data)

def query_api_batch(applications_df, batch_size=100):
    """
    Query the victim's API using HTTP requests.
    Uses batch endpoint for efficiency.
    """
    all_decisions = []
    blocked = False
    
    for i in range(0, len(applications_df), batch_size):
        batch = applications_df.iloc[i:i+batch_size]
        
        # Convert to list of dicts for JSON
        payload = {
            "applications": batch.to_dict(orient='records')
        }
        
        response = requests.post(
            f"{API_BASE_URL}/batch_predict",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            results = response.json()['results']
            decisions = [r['decision_code'] for r in results]
            all_decisions.extend(decisions)
        elif response.status_code == 429:
            # Rate limited / blocked - stop and return what we have
            print(f"\n\n   {RED}🛡️  BLOCKED by API rate limiting!{RESET}")
            print(f"   {YELLOW}   Attack stopped after {len(all_decisions)} queries{RESET}")
            blocked = True
            break
        else:
            raise Exception(f"API error: {response.text}")
    
    return np.array(all_decisions), blocked

def query_api_single(application):
    """Query the API for a single application."""
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=application,
        timeout=10
    )
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API error: {response.text}")

def query_attack():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}{RED}🔓 STEP 2: HTTP API Query Attack{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")
    
    # ========================================
    # PHASE 0: Reconnaissance - Discover API
    # ========================================
    print(f"{BOLD}{CYAN}🔎 PHASE 0: API Reconnaissance{RESET}")
    print(f"   Discovering API endpoints and parameters...\n")
    
    try:
        # Check API health
        health = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print(f"{RED}   ❌ API not responding. Start 1b_api_server.py first!{RESET}")
            return None
        
        # Get API info (attacker discovers features from docs)
        info = requests.get(f"{API_BASE_URL}/api_info", timeout=5).json()
        print(f"   ✅ Target API: {info['name']}")
        print(f"   ✅ Discovered features: {len(info['required_features'])}")
        print(f"   ✅ Possible decisions: {info['possible_decisions']}")
        
    except requests.exceptions.ConnectionError:
        print(f"{RED}   ❌ Cannot connect to API at {API_BASE_URL}{RESET}")
        print(f"{YELLOW}   ➡️  Run '1b_api_server.py' in another terminal first!{RESET}")
        return None
    
    # ========================================
    # PHASE 1: Generate Probe Applications
    # ========================================
    print(f"\n{BOLD}{CYAN}📡 PHASE 1: Generating Synthetic Loan Applications{RESET}")
    print(f"   Attacker creates fake loan applications to probe the API...\n")
    
    n_queries = 8000
    probe_df = generate_probe_applications(n_samples=n_queries, random_state=123)
    
    # Show sample probes
    print(f"   {BOLD}Sample Probe Applications:{RESET}")
    for i in range(3):
        row = probe_df.iloc[i]
        print(f"   Probe {i+1}: Income=${row['annual_income']:,.0f}, "
              f"Credit={row['credit_score']:.0f}, "
              f"Loan=${row['loan_amount']:,.0f}")
    
    print(f"\n   • Generated {n_queries} synthetic loan applications")
    print(f"   • Features: {len(FEATURE_NAMES)} (discovered from /api_info)")
    
    # ========================================
    # PHASE 2: Query the API via HTTP
    # ========================================
    print(f"\n{BOLD}{CYAN}🌐 PHASE 2: Sending HTTP Requests to API{RESET}")
    print(f"   POST {API_BASE_URL}/batch_predict\n")
    
    start_time = time.time()
    
    # Query the API using HTTP requests
    print(f"   Sending requests", end="", flush=True)
    stolen_labels, was_blocked = query_api_batch(probe_df, batch_size=100)
    
    query_time = time.time() - start_time
    
    # Handle blocked case
    if was_blocked:
        print(f"\n   {RED}❌ Attack was BLOCKED by security controls!{RESET}")
        print(f"   ⏱️  Time before block: {query_time:.3f} seconds")
        
        if len(stolen_labels) < 100:
            print(f"\n   {GREEN}🛡️  DEFENSE SUCCESSFUL!{RESET}")
            print(f"   Insufficient data collected ({len(stolen_labels)} samples)")
            print(f"   Cannot train a useful surrogate model.")
            print(f"\n{BOLD}{'='*60}{RESET}")
            print(f"{GREEN}✅ Model stealing attack PREVENTED!{RESET}")
            print(f"{BOLD}{'='*60}{RESET}\n")
            return
        else:
            print(f"\n   {YELLOW}⚠️  Partial data collected: {len(stolen_labels)} samples{RESET}")
            print(f"   Attempting to train with limited (possibly noisy) data...")
            # Trim probe_df to match stolen labels
            probe_df = probe_df.iloc[:len(stolen_labels)]
    else:
        print(f" ✅")
    
    # Count decisions
    label_counts = np.bincount(stolen_labels, minlength=3)
    
    print(f"\n   ✅ Collected {len(stolen_labels)} API responses")
    print(f"   ⏱️  Query time: {query_time:.3f} seconds")
    print(f"\n   {BOLD}Stolen API Responses:{RESET}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"   • {name}: {label_counts[i]} decisions")
    
    # ========================================
    # PHASE 3: Train Surrogate Model
    # ========================================
    print(f"\n{BOLD}{CYAN}🧠 PHASE 3: Training Surrogate Loan Model{RESET}")
    print(f"   Using stolen API responses to clone FinTech's algorithm...\n")
    
    start_time = time.time()
    
    # Train a surrogate model using stolen knowledge
    surrogate_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    # Train on stolen data (raw features, no scaling needed)
    surrogate_model.fit(probe_df.values, stolen_labels)
    
    train_time = time.time() - start_time
    
    print(f"   ✅ Surrogate loan model trained!")
    print(f"   ⏱️  Training time: {train_time:.3f} seconds")
    
    # Save the stolen model
    joblib.dump(surrogate_model, 'models/stolen_model.joblib')
    
    if was_blocked:
        print(f"\n{YELLOW}⚠️  Partial Attack (was blocked){RESET}")
        print(f"   Attacker trained model with only {len(stolen_labels)} samples")
        print(f"   Model quality likely degraded due to:")
        print(f"   • Insufficient training data")
        print(f"   • Possible differential privacy noise in responses")
    else:
        print(f"\n{GREEN}💀 Attack Successful via HTTP API!{RESET}")
        print(f"   Attacker now has their own loan approval model!")
    
    print(f"   Saved to: models/stolen_model.joblib")
    
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{YELLOW}➡️  Next: Run '3_compare_models.py' to see the damage!{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

if __name__ == "__main__":
    query_attack()
