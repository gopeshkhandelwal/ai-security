#!/usr/bin/env python3
"""
Step 1: Create the Victim's Proprietary Model

This script simulates a company's valuable ML model - their intellectual property.
The model is trained on proprietary data and represents significant investment.

Scenario: A FinTech company's LOAN APPROVAL MODEL
- Trained on years of customer data
- Predicts: Approved, Denied, or Manual Review
- Exposed as an API for partners/applications
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Color codes for terminal output
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'

# Feature names for the loan approval model
FEATURE_NAMES = [
    'annual_income',        # Annual income in $
    'credit_score',         # Credit score (300-850)
    'debt_to_income',       # Debt-to-income ratio
    'employment_years',     # Years at current job
    'loan_amount',          # Requested loan amount
    'loan_term_months',     # Loan term in months
    'num_credit_lines',     # Number of credit lines
    'num_late_payments',    # Late payments in last 2 years
    'home_ownership',       # 0=Rent, 1=Mortgage, 2=Own
    'account_age_months'    # Age of oldest account
]

CLASS_NAMES = ['DENIED', 'MANUAL_REVIEW', 'APPROVED']

def generate_loan_data(n_samples=1000, random_state=42):
    """
    Generate realistic-looking loan application data.
    This simulates a FinTech company's proprietary customer dataset.
    """
    np.random.seed(random_state)
    
    # Generate realistic feature distributions
    data = {
        'annual_income': np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples),  # ~$50k median
        'credit_score': np.clip(np.random.normal(680, 80, n_samples), 300, 850),
        'debt_to_income': np.clip(np.random.exponential(0.25, n_samples), 0, 1),
        'employment_years': np.clip(np.random.exponential(5, n_samples), 0, 40),
        'loan_amount': np.random.lognormal(mean=9.5, sigma=0.8, size=n_samples),  # ~$13k median
        'loan_term_months': np.random.choice([12, 24, 36, 48, 60, 72], n_samples),
        'num_credit_lines': np.clip(np.random.poisson(5, n_samples), 0, 20),
        'num_late_payments': np.clip(np.random.poisson(1, n_samples), 0, 10),
        'home_ownership': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.45, 0.15]),
        'account_age_months': np.clip(np.random.exponential(80, n_samples), 6, 360)
    }
    
    df = pd.DataFrame(data)
    
    # Create labels based on realistic business rules (company's secret sauce)
    # This scoring logic represents the company's proprietary IP
    score = (
        (df['credit_score'] - 300) / 550 * 30 +                    # Credit score (30%)
        np.clip(df['annual_income'] / 200000, 0, 1) * 20 +         # Income (20%)
        (1 - df['debt_to_income']) * 15 +                           # Low DTI (15%)
        np.clip(df['employment_years'] / 10, 0, 1) * 10 +          # Employment (10%)
        (1 - df['num_late_payments'] / 10) * 15 +                   # Payment history (15%)
        (df['home_ownership'] / 2) * 5 +                            # Home ownership (5%)
        np.clip(df['account_age_months'] / 120, 0, 1) * 5           # Account age (5%)
    )
    
    # Add some noise to make it realistic
    score += np.random.normal(0, 5, n_samples)
    
    # Classify into 3 categories
    labels = np.where(score < 40, 0,           # DENIED
              np.where(score < 65, 1,          # MANUAL_REVIEW  
                       2))                      # APPROVED
    
    return df.values, labels, df

def create_victim_model():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}🏢 STEP 1: Creating Proprietary Loan Approval Model{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")
    
    # Generate proprietary training data (company's secret dataset)
    print(f"{YELLOW}📊 Generating proprietary customer loan data...{RESET}")
    X, y, df = generate_loan_data(n_samples=1000, random_state=42)
    
    # Show sample data
    print(f"\n   {BOLD}Sample Customer Records:{RESET}")
    sample_df = pd.DataFrame(X[:3], columns=FEATURE_NAMES)
    for i, row in sample_df.iterrows():
        print(f"   Customer {i+1}: Income=${row['annual_income']:,.0f}, "
              f"Credit={row['credit_score']:.0f}, "
              f"DTI={row['debt_to_income']:.1%}, "
              f"Loan=${row['loan_amount']:,.0f}")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n   ✅ Training samples: {len(X_train)}")
    print(f"   ✅ Test samples: {len(X_test)} (kept secret)")
    print(f"   ✅ Features: {len(FEATURE_NAMES)}")
    print(f"   ✅ Classes: {CLASS_NAMES}")
    
    # Train the proprietary model (company's valuable IP)
    print(f"\n{YELLOW}🔧 Training proprietary neural network...{RESET}")
    
    victim_model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),  # Secret architecture
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    victim_model.fit(X_train, y_train)
    
    # Evaluate accuracy
    train_acc = victim_model.score(X_train, y_train)
    test_acc = victim_model.score(X_test, y_test)
    
    print(f"\n{GREEN}✅ Proprietary Model Trained Successfully!{RESET}")
    print(f"\n{BOLD}📈 Model Performance (SECRET - only known to owner):{RESET}")
    print(f"   • Training Accuracy: {train_acc:.2%}")
    print(f"   • Test Accuracy: {test_acc:.2%}")
    print(f"   • Architecture: {victim_model.hidden_layer_sizes} (SECRET)")
    
    # Save the proprietary model
    os.makedirs('models', exist_ok=True)
    joblib.dump(victim_model, 'models/proprietary_model.joblib')
    
    print(f"\n{BLUE}💾 Saved:{RESET}")
    print(f"   • models/proprietary_model.joblib")
    
    print(f"\n{BOLD}{RED}⚠️  SCENARIO:{RESET}")
    print(f"   FinTech Corp's Loan Approval API is deployed for partners.")
    print(f"   Partners can submit loan applications and get decisions:")
    print(f"   • Input: Customer financial data (10 features)")
    print(f"   • Output: APPROVED / DENIED / MANUAL_REVIEW")
    print(f"   ")
    print(f"   {BOLD}Partners CANNOT access:{RESET}")
    print(f"   • The proprietary scoring algorithm")
    print(f"   • The model weights or architecture")
    print(f"   • Historical customer data used for training")
    
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{YELLOW}➡️  Next: Run '2_query_attack.py' to steal this model!{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

if __name__ == "__main__":
    create_victim_model()
