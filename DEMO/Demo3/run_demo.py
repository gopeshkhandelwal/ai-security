#!/usr/bin/env python3
"""
Demo 5: Model Stealing via Query Attack - ALL IN ONE

Run this single script to see the entire attack demonstration.
Executes in less than 1 minute on CPU.

Scenario: Stealing a FinTech company's Loan Approval Model

MITRE ATLAS ATT&CK Techniques:
- AML.T0044: Full ML Model Access
- AML.T0024: Exfiltration via ML Inference API
- AML.T0035: ML Model Inference API Access
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Color codes
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RED = '\033[91m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
BOLD = '\033[1m'

FEATURE_NAMES = [
    'annual_income', 'credit_score', 'debt_to_income', 'employment_years',
    'loan_amount', 'loan_term_months', 'num_credit_lines', 'num_late_payments',
    'home_ownership', 'account_age_months'
]

CLASS_NAMES = ['DENIED', 'MANUAL_REVIEW', 'APPROVED']

def print_banner():
    banner = f"""
{BOLD}{RED}
╔═══════════════════════════════════════════════════════════════════╗
║         🔓 LOAN MODEL STEALING VIA QUERY ATTACK 🔓                ║
║                                                                   ║
║   MITRE ATLAS: AML.T0044, AML.T0024, AML.T0035                   ║
║                                                                   ║
║   Scenario: Attacker steals FinTech Corp's proprietary           ║
║   loan approval algorithm using only API query access.            ║
╚═══════════════════════════════════════════════════════════════════╝
{RESET}"""
    print(banner)

def generate_loan_data(n_samples=1000, random_state=42):
    """Generate realistic loan application data."""
    np.random.seed(random_state)
    
    data = {
        'annual_income': np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples),
        'credit_score': np.clip(np.random.normal(680, 80, n_samples), 300, 850),
        'debt_to_income': np.clip(np.random.exponential(0.25, n_samples), 0, 1),
        'employment_years': np.clip(np.random.exponential(5, n_samples), 0, 40),
        'loan_amount': np.random.lognormal(mean=9.5, sigma=0.8, size=n_samples),
        'loan_term_months': np.random.choice([12, 24, 36, 48, 60, 72], n_samples),
        'num_credit_lines': np.clip(np.random.poisson(5, n_samples), 0, 20),
        'num_late_payments': np.clip(np.random.poisson(1, n_samples), 0, 10),
        'home_ownership': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.45, 0.15]),
        'account_age_months': np.clip(np.random.exponential(80, n_samples), 6, 360)
    }
    
    df = pd.DataFrame(data)
    
    # Proprietary scoring logic (FinTech's secret sauce)
    score = (
        (df['credit_score'] - 300) / 550 * 30 +
        np.clip(df['annual_income'] / 200000, 0, 1) * 20 +
        (1 - df['debt_to_income']) * 15 +
        np.clip(df['employment_years'] / 10, 0, 1) * 10 +
        (1 - df['num_late_payments'] / 10) * 15 +
        (df['home_ownership'] / 2) * 5 +
        np.clip(df['account_age_months'] / 120, 0, 1) * 5
    )
    score += np.random.normal(0, 5, n_samples)
    
    labels = np.where(score < 40, 0, np.where(score < 65, 1, 2))
    
    return df.values, labels, df

def generate_probe_applications(n_samples=2000, random_state=123):
    """Attacker generates fake loan applications to probe the API."""
    np.random.seed(random_state)
    
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

def run_demo():
    print_banner()
    total_start = time.time()
    
    # ========================================
    # STEP 1: Create Victim's Loan Model
    # ========================================
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}{BLUE}🏢 STEP 1: FinTech Corp's Proprietary Loan Model{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")
    
    print(f"\n{YELLOW}📊 Generating proprietary customer loan data...{RESET}")
    X, y, df = generate_loan_data(n_samples=1000, random_state=42)
    
    # Show sample data
    print(f"\n   {BOLD}Sample Customer Records:{RESET}")
    for i in range(3):
        print(f"   Customer {i+1}: Income=${df.iloc[i]['annual_income']:,.0f}, "
              f"Credit={df.iloc[i]['credit_score']:.0f}, "
              f"Loan=${df.iloc[i]['loan_amount']:,.0f}")
    
    # Normalize and split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    print(f"\n   ✅ Training: {len(X_train)} customers | Test: {len(X_test)} customers")
    
    print(f"\n{YELLOW}🔧 Training proprietary loan approval model...{RESET}")
    victim_model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu', solver='adam',
        max_iter=500, random_state=42, early_stopping=True
    )
    victim_model.fit(X_train, y_train)
    
    victim_train_acc = victim_model.score(X_train, y_train)
    victim_test_acc = victim_model.score(X_test, y_test)
    
    print(f"\n{GREEN}✅ FinTech's Loan Model Ready!{RESET}")
    print(f"   • Architecture: (64, 32, 16) hidden layers {YELLOW}[SECRET]{RESET}")
    print(f"   • Training Accuracy: {victim_train_acc:.2%}")
    print(f"   • Test Accuracy: {victim_test_acc:.2%}")
    print(f"\n{RED}   ⚠️  Model deployed as Loan Approval API...{RESET}")
    
    # ========================================
    # STEP 2: Query Attack
    # ========================================
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}{RED}🔓 STEP 2: Attacker's Query Attack{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")
    
    attack_start = time.time()
    
    # Phase 1: Generate fake loan applications
    print(f"\n{CYAN}📡 Phase 1: Generating fake loan applications...{RESET}")
    n_queries = 2000
    probe_df = generate_probe_applications(n_samples=n_queries, random_state=123)
    
    print(f"   {BOLD}Sample Probes:{RESET}")
    for i in range(2):
        row = probe_df.iloc[i]
        print(f"   Probe {i+1}: Income=${row['annual_income']:,.0f}, "
              f"Credit={row['credit_score']:.0f}, Loan=${row['loan_amount']:,.0f}")
    print(f"   ... {n_queries} total applications")
    
    # Phase 2: Query victim API
    print(f"\n{CYAN}🔍 Phase 2: Querying FinTech's Loan API...{RESET}")
    X_probe_scaled = scaler.transform(probe_df.values)
    stolen_labels = victim_model.predict(X_probe_scaled)
    
    label_counts = np.bincount(stolen_labels, minlength=3)
    print(f"   Collected decisions: DENIED={label_counts[0]}, "
          f"MANUAL={label_counts[1]}, APPROVED={label_counts[2]}")
    
    # Phase 3: Train surrogate
    print(f"\n{CYAN}🧠 Phase 3: Training stolen loan model...{RESET}")
    surrogate_model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu', solver='adam',
        max_iter=300, random_state=42, early_stopping=True
    )
    surrogate_model.fit(X_probe_scaled, stolen_labels)
    print(f"   ✅ Surrogate model trained!")
    
    attack_time = time.time() - attack_start
    
    # ========================================
    # STEP 3: Attack Results
    # ========================================
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}{MAGENTA}📊 STEP 3: Attack Results{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")
    
    victim_preds = victim_model.predict(X_test)
    stolen_preds = surrogate_model.predict(X_test)
    
    stolen_acc = accuracy_score(y_test, stolen_preds)
    fidelity = accuracy_score(victim_preds, stolen_preds)
    theft_rate = (stolen_acc / victim_test_acc) * 100
    
    print(f"""
   ┌────────────────────────────────────────────────┐
   │  {BOLD}LOAN MODEL THEFT ANALYSIS{RESET}                      │
   ├────────────────────────────────────────────────┤
   │  FinTech Model Accuracy:   {victim_test_acc:>6.2%}            │
   │  {RED}Stolen Model Accuracy:    {stolen_acc:>6.2%}{RESET}            │
   │  Decision Agreement:       {fidelity:>6.2%}            │
   │  {RED}Theft Success Rate:       {theft_rate:>5.1f}%{RESET}            │
   │  Attack Duration:          {attack_time:>5.2f}s             │
   └────────────────────────────────────────────────┘
""")
    
    if fidelity > 0.85:
        print(f"   {BOLD}{RED}🚨 HIGH FIDELITY THEFT SUCCESSFUL!{RESET}")
        print(f"   Attacker can now approve/deny loans like FinTech!")
    
    # ========================================
    # Attack Economics
    # ========================================
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}{GREEN}💰 Attack Economics{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")
    
    print(f"""
   {BOLD}FinTech Corp's Investment:{RESET}
   • Customer data (years of loans)    💲💲💲💲
   • Data science team                 💲💲💲
   • Regulatory compliance             💲💲
   • Total: ~$1,000,000+
   
   {BOLD}Attacker's Cost:{RESET}
   • {n_queries} API queries           ~$0
   • CPU training time                 ~$0
   
   {BOLD}{RED}Result: Stole loan algorithm for FREE!{RESET}
""")
    
    # ========================================
    # Defenses
    # ========================================
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}{GREEN}🛡️  Defense Recommendations{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")
    
    print(f"""
   1. {BOLD}Rate Limiting{RESET} - Max queries per user/hour
   2. {BOLD}Query Monitoring{RESET} - Detect synthetic inputs  
   3. {BOLD}Output Perturbation{RESET} - Add noise to decisions
   4. {BOLD}Model Watermarking{RESET} - Track stolen models
   5. {BOLD}API Authentication{RESET} - Audit all queries
""")
    
    # Summary
    total_time = time.time() - total_start
    
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}{RED}🎯 KEY TAKEAWAY{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")
    print(f"""
   {BOLD}Query access to ML APIs = IP theft risk!{RESET}
   
   • Attacker used {n_queries} fake loan applications
   • Stole {fidelity:.0%} of FinTech's decision logic
   • No ML expertise needed - just copy the outputs
   
   {BOLD}Protect ML APIs like you protect trade secrets.{RESET}
   
   ⏱️  Total demo time: {total_time:.2f} seconds
""")

if __name__ == "__main__":
    run_demo()
