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
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
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
    data = {
        'annual_income': np.random.uniform(20000, 300000, n_samples),
        'credit_score': np.random.uniform(300, 850, n_samples),
        'debt_to_income': np.random.uniform(0, 0.8, n_samples),
        'employment_years': np.random.uniform(0, 30, n_samples),
        'loan_amount': np.random.uniform(1000, 100000, n_samples),
        'loan_term_months': np.random.choice([12, 24, 36, 48, 60, 72], n_samples),
        'num_credit_lines': np.random.randint(0, 15, n_samples),
        'num_late_payments': np.random.randint(0, 8, n_samples),
        'home_ownership': np.random.choice([0, 1, 2], n_samples),
        'account_age_months': np.random.uniform(6, 300, n_samples)
    }
    return pd.DataFrame(data)

def query_api_batch(applications_df, batch_size=100):
    """
    Query the victim's API using HTTP requests.
    Uses batch endpoint for efficiency.
    """
    all_decisions = []
    
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
        else:
            raise Exception(f"API error: {response.text}")
    
    return np.array(all_decisions)

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
    
    n_queries = 2000
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
    stolen_labels = query_api_batch(probe_df, batch_size=200)
    print(f" ✅")
    
    query_time = time.time() - start_time
    
    # Count decisions
    label_counts = np.bincount(stolen_labels, minlength=3)
    
    print(f"\n   ✅ Sent {n_queries} HTTP requests")
    print(f"   ✅ Collected {n_queries} API responses")
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
    surrogate_model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42,
        early_stopping=True
    )
    
    # Train on stolen data (raw features, no scaling needed)
    surrogate_model.fit(probe_df.values, stolen_labels)
    
    train_time = time.time() - start_time
    
    print(f"   ✅ Surrogate loan model trained!")
    print(f"   ⏱️  Training time: {train_time:.3f} seconds")
    
    # Save the stolen model
    joblib.dump(surrogate_model, 'models/stolen_model.joblib')
    
    print(f"\n{GREEN}💀 Attack Successful via HTTP API!{RESET}")
    print(f"   Attacker now has their own loan approval model!")
    print(f"   Saved to: models/stolen_model.joblib")
    
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{YELLOW}➡️  Next: Run '3_compare_models.py' to see the damage!{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

if __name__ == "__main__":
    query_attack()
